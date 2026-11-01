"""Independent electromagnetic port surrogate and fail-closed qualification.

This is a learned 3D port model, not a Maxwell solver. No eligible headset
weights ship with MicroScan. Synthetic qualification is only a mechanism test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

DOMAINS = {"raw_s", "calibrated_delta_s", "decluttered_target"}


def normalize_domain(domain):
    # Explicit legacy spelling alias; physical domains are never interchanged.
    return "calibrated_delta_s" if domain == "calibrated_delta" else domain
QUALIFICATION_VERSION = 1


def _digest(value):
    if torch.is_tensor(value):
        a = value.detach().cpu().contiguous()
        return {"shape": list(a.shape), "values": a.tolist()}
    if isinstance(value, Mapping):
        return {k: _digest(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_digest(v) for v in value]
    return value


def _canonical_context_value(key, value):
    dimensions = {"tx_positions_mm": 2, "rx_positions_mm": 2, "source_excitation": 2,
                  "tx_orientations": 2, "rx_orientations": 2, "calibration_features": 1}
    if key in dimensions:
        x = torch.as_tensor(value).detach().cpu().to(torch.float64)
        rank = dimensions[key]
        if x.ndim == rank+1:
            if not torch.allclose(x, x[:1].expand_as(x), atol=1e-6, rtol=0):
                raise ValueError("a batch cannot mix independently bound hardware/source/calibration contexts")
            x = x[0]
        if x.ndim != rank or not torch.isfinite(x).all():
            raise ValueError(f"invalid qualification context {key}")
        # Stable across float32 export versus JSON double and singleton batching.
        x = torch.round(x*1e6)/1e6
        return x.tolist()
    if isinstance(value, (list, tuple)) and value and all(v == value[0] for v in value):
        return value[0]
    return _digest(value)


def context_hash(context):
    """Bind fixed headset/source/calibration context, invariant to batch repeats.

    Geometry descriptors are checked against separately predeclared qualified
    ranges, rather than pinning a new qualification to each object's geometry.
    """
    keys = ("layout_id", "calibration_id", "grid_id", "material_library_id", "source_id",
            "tx_positions_mm", "rx_positions_mm", "tx_orientations", "rx_orientations",
            "source_excitation", "calibration_features")
    bound = {k: _canonical_context_value(k, context[k]) for k in keys if k in context}
    bound["geometry_features_present"] = "geometry_features" in context
    return hashlib.sha256(json.dumps(bound, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _geometry_in_bounds(context, bounds):
    if "geometry_features" not in context:
        return bounds is None
    if bounds is None:
        return False
    values = torch.as_tensor(context["geometry_features"]).detach().cpu().to(torch.float64)
    ranges = torch.as_tensor(bounds, dtype=torch.float64)
    if values.ndim not in (1, 2) or values.shape[-1] != 4 or ranges.shape != (4, 2):
        return False
    return bool(torch.isfinite(values).all() and torch.isfinite(ranges).all()
                and (ranges[:, 0] <= ranges[:, 1]).all()
                and (values >= ranges[:, 0]-1e-6).all()
                and (values <= ranges[:, 1]+1e-6).all())


def weights_hash(operator):
    digest = hashlib.sha256()
    for key, value in sorted(operator.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class HeadsetForwardOperator(nn.Module):
    """Shared 3D volume encoder and frequency/source/port-conditioned decoder.

    properties: [B,2,Z,Y,X], or [B,2,Q,Z,Y,X] with properties at each query.
    output: [B,2,Q,Rx,Tx], real and imaginary, in the declared signal domain.
    Physical epsilon/conductivity are scaled by fixed *saved* encoder scales.
    Current measurements and labels cannot enter ``context`` as shortcuts.
    """
    def __init__(self, width=16, latent_dim=64, domain="calibrated_delta",
                 property_scales=(44.0, 0.8), property_offsets=(1.0, 0.0),
                 context_dim=8):
        super().__init__()
        domain = normalize_domain(domain)
        if domain not in DOMAINS:
            raise ValueError(f"unknown signal domain {domain}")
        if context_dim != 8:
            raise ValueError("context_dim is 8: four geometry plus four calibration features")
        self.domain = domain
        self.calls = 0
        self.register_buffer("property_scales", torch.tensor(property_scales).view(1, 2, 1, 1, 1))
        self.register_buffer("property_offsets", torch.tensor(property_offsets).view(1, 2, 1, 1, 1))
        def block(i, o):
            return nn.Sequential(nn.Conv3d(i, o, 3, stride=2, padding=1),
                                 nn.GroupNorm(1, o), nn.SiLU())
        self.encoder = nn.Sequential(block(2, width), block(width, width*2),
                                     block(width*2, width*4), nn.AdaptiveAvgPool3d(1),
                                     nn.Flatten(), nn.Linear(width*4, latent_dim), nn.SiLU())
        # position 6, orientation 6, source RI 2, freq 1, geometry/calibration 8,
        # geometry/calibration validity 2: 25 explicit physical context values.
        self.decoder = nn.Sequential(nn.Linear(latent_dim+25, latent_dim), nn.SiLU(),
                                     nn.Linear(latent_dim, latent_dim), nn.SiLU(),
                                     nn.Linear(latent_dim, 2))
        self.qualification_report = None

    @staticmethod
    def _batch(value, batch, device, dtype, dimensions):
        x = torch.as_tensor(value, device=device, dtype=dtype)
        if x.ndim == dimensions-1:
            x = x.unsqueeze(0).expand(batch, *x.shape)
        if x.ndim != dimensions or x.shape[0] != batch:
            raise ValueError("incompatible physical context batch/axes")
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite physical context")
        return x

    def forward(self, properties, frequencies_ghz, context):
        forbidden = {"signals", "measurements", "observations", "target", "labels", "raw_s"}
        if forbidden.intersection(context):
            raise ValueError("Phi context must not contain current observations or labels")
        b = properties.shape[0]
        device, dtype = properties.device, properties.dtype
        frequencies = torch.as_tensor(frequencies_ghz, device=device, dtype=dtype)
        if frequencies.ndim == 1:
            frequencies = frequencies[None].expand(b, -1)
        if frequencies.ndim != 2 or frequencies.shape[0] != b or not torch.isfinite(frequencies).all():
            raise ValueError("query frequencies must be finite [Q] or [B,Q] GHz")
        q = frequencies.shape[1]
        if properties.ndim == 5:
            if not context.get("constant_spectrum", False) and q > 1:
                raise ValueError("multiple frequencies require spectral properties or explicit constant_spectrum")
            volumes = properties[:, None].expand(-1, q, -1, -1, -1, -1)
        elif properties.ndim == 6 and properties.shape[1:3] == (2, q):
            volumes = properties.permute(0, 2, 1, 3, 4, 5)
        else:
            raise ValueError("properties axes must be B,2,Z,Y,X or B,2,Q,Z,Y,X")
        if not torch.isfinite(volumes).all():
            raise ValueError("nonfinite physical property input")
        tx = self._batch(context["tx_positions_mm"], b, device, dtype, 3)
        rx = self._batch(context["rx_positions_mm"], b, device, dtype, 3)
        source = self._batch(context["source_excitation"], b, device, dtype, 3)
        t, r = tx.shape[1], rx.shape[1]
        if tx.shape[-1] != 3 or rx.shape[-1] != 3 or source.shape[1:] != (t, 2):
            raise ValueError("ports need XYZ mm positions and source [Tx,2] RI excitation")
        txo = self._batch(context.get("tx_orientations", torch.zeros(t, 3)), b, device, dtype, 3)
        rxo = self._batch(context.get("rx_orientations", torch.zeros(r, 3)), b, device, dtype, 3)
        extras = []
        for name in ("geometry_features", "calibration_features"):
            present = name in context
            feature = self._batch(context.get(name, torch.zeros(4)), b, device, dtype, 2)
            if feature.shape[1] != 4:
                raise ValueError(f"{name} requires 4 recorded features")
            extras.extend([feature, torch.full((b, 1), float(present), device=device, dtype=dtype)])
        extra = torch.cat(extras, -1)
        # Encode each frequency's physical volume once, never once per antenna.
        normalized = (volumes.reshape(b*q, 2, *volumes.shape[-3:]) - self.property_offsets) / self.property_scales
        latent = self.encoder(normalized).reshape(b, q, -1)
        def expand(x):
            return x[:, None, None].expand(b, q, r, t, -1)
        tx_features = expand(torch.cat((tx/100.0, txo, source), -1))
        rx_features = torch.cat((rx/100.0, rxo), -1)[:, None, :, None].expand(b, q, r, t, -1)
        freq_features = frequencies[:, :, None, None, None].expand(b, q, r, t, 1)
        extra_features = extra[:, None, None, None].expand(b, q, r, t, -1)
        feature = latent[:, :, None, None].expand(b, q, r, t, -1)
        inputs = torch.cat((feature, tx_features, rx_features, freq_features, extra_features), -1)
        self.calls += 1
        return self.decoder(inputs).permute(0, 4, 1, 2, 3).contiguous()


def masked_complex_loss(predicted, observed, mask, noise_scale=None, covariance=None,
                        floor=1e-8):
    """Real/imag quadratic loss; select observed entries *before* whitening.

    Covariance is real [2*M,2*M] or [B,2*M,2*M], with flattening of
    [2,F,R,T]. A principal observed submatrix is used; unobserved covariance
    entries may be NaN and do not enter Cholesky or normalization.
    """
    if predicted.shape != observed.shape or predicted.ndim != 5 or predicted.shape[1] != 2:
        raise ValueError("complex signals require matching [B,2,F,R,T]")
    support = torch.broadcast_to(mask.bool(), predicted.shape)
    # where, rather than multiplication: 0*NaN is still NaN.
    p = torch.where(support, predicted, torch.zeros_like(predicted))
    o = torch.where(support, observed, torch.zeros_like(observed))
    if not torch.isfinite(p).all() or not torch.isfinite(o).all():
        raise ValueError("observed signal entries must be finite")
    residual = p-o
    if noise_scale is not None:
        scale = torch.broadcast_to(torch.as_tensor(noise_scale, device=p.device, dtype=p.dtype), p.shape)
        scale = torch.where(support, scale, torch.ones_like(scale))
        if not torch.isfinite(scale).all() or (scale[support] < 0).any():
            raise ValueError("invalid observed noise scale")
        residual = residual / scale.clamp_min(floor)
    total = residual.sum()*0.0
    count = 0
    for i in range(p.shape[0]):
        valid = support[i].reshape(-1)
        values = residual[i].reshape(-1)[valid]
        if not values.numel():
            continue
        if covariance is not None:
            cov = torch.as_tensor(covariance, device=p.device, dtype=p.dtype)
            cov = cov[i] if cov.ndim == 3 else cov
            if cov.shape != (valid.numel(), valid.numel()):
                raise ValueError("covariance must use flattened real/imag signal axes")
            cov = cov[valid][:, valid]
            if not torch.isfinite(cov).all():
                raise ValueError("nonfinite observed covariance")
            cov = (cov+cov.T)*.5
            chol = torch.linalg.cholesky(cov + floor*torch.eye(len(values), device=p.device, dtype=p.dtype))
            values = torch.linalg.solve_triangular(chol, values[:, None], upper=False).squeeze(-1)
        total = total+values.square().sum()
        count += values.numel()
    return total / max(count, 1)


@dataclass
class QualificationThresholds:
    full_relative_rmse: float = 0.05
    differential_relative_rmse: float = 0.10
    noise_relative_rmse: float = 1.0
    worst_channel_relative_rmse: float = 0.20
    roi_relative_rmse: float = 0.10


def qualification_report(operator, metrics, context, frequencies_ghz, *,
                         training_groups, validation_groups, coverage,
                         thresholds=None, scientific=True, property_bounds=None,
                         geometry_feature_bounds=None):
    """Build independently measured gate report; no fabricated missing metrics.

    ``coverage`` lists held-out region/source/contrast-sign cases actually tested.
    Caller supplies metrics measured on the independent validation set. The
    evaluator below computes signal errors; cohort coverage remains export data.
    """
    thresholds = thresholds or QualificationThresholds()
    if isinstance(thresholds, dict):
        thresholds = QualificationThresholds(**thresholds)
    required_coverage = {"healthy", "lesion", "positive_change", "negative_change", "sources", "regions"}
    reasons = []
    if not scientific:
        reasons.append("synthetic_fixture_not_scientific_qualification")
    if not training_groups or not validation_groups or set(training_groups) & set(validation_groups):
        reasons.append("missing_or_overlapping_independent_groups")
    if not required_coverage.issubset(set(coverage)):
        reasons.append("incomplete_validation_coverage")
    for name, limit in asdict(thresholds).items():
        value = metrics.get(name)
        if value is None or not math.isfinite(float(value)) or float(value) > limit:
            reasons.append(f"metric_gate_failed:{name}")
    if property_bounds is None:
        reasons.append("missing_property_domain_bounds")
    if not _geometry_in_bounds(context, geometry_feature_bounds):
        reasons.append("missing_or_out_of_range_geometry_feature_bounds")
    report = {"version": QUALIFICATION_VERSION, "qualified": not reasons,
              "scientific": bool(scientific), "fixture_eligible": (not scientific and
                all(not r.startswith("metric_gate_failed") for r in reasons)),
              "reasons": reasons, "domain": operator.domain,
              "weights_sha256": weights_hash(operator), "context_sha256": context_hash(context),
              "frequencies_ghz": torch.as_tensor(frequencies_ghz).tolist(),
              "thresholds": asdict(thresholds),
              "metrics": {k: (float(v) if v is not None and math.isfinite(float(v)) else None) for k,v in metrics.items()},
              "training_groups": list(training_groups), "validation_groups": list(validation_groups),
              "coverage": sorted(coverage), "property_bounds": property_bounds,
              "geometry_feature_bounds": geometry_feature_bounds}
    operator.qualification_report = report
    return report


def require_qualified_operator(operator, report=None, context=None, domain=None, allow_fixture=False):
    report = report or getattr(operator, "qualification_report", None)
    if report is None or report.get("version") != QUALIFICATION_VERSION:
        raise ValueError("Phi requires a versioned independent qualification report")
    eligible = report.get("qualified") and report.get("scientific")
    if allow_fixture:
        eligible = eligible or report.get("fixture_eligible", False)
    if not eligible:
        raise ValueError("Phi is unqualified: " + ", ".join(report.get("reasons", [])))
    if report.get("weights_sha256") != weights_hash(operator):
        raise ValueError("Phi weights do not match qualification report")
    domain = normalize_domain(domain)
    if domain is not None and (domain != operator.domain or normalize_domain(report.get("domain")) != domain):
        raise ValueError("Phi measurement domain mismatch")
    if context is not None:
        if report.get("context_sha256") != context_hash(context):
            raise ValueError("Phi hardware/source/calibration context mismatch")
        if not _geometry_in_bounds(context, report.get("geometry_feature_bounds")):
            raise ValueError("Phi queried outside qualified geometry feature domain")
    return report


def freeze_operator(operator):
    operator.eval()
    for p in operator.parameters():
        p.requires_grad_(False)
        p.grad = None
    return operator


class PhysicsConsistency(nn.Module):
    """Training-only B loss, retaining inverse input gradients through frozen Phi."""
    def __init__(self, operator, qualification=None, domain="calibrated_delta",
                 require_qualified=True, allow_fixture=False):
        super().__init__()
        domain = normalize_domain(domain)
        if domain not in DOMAINS:
            raise ValueError("unknown observation domain")
        self.operator = freeze_operator(operator)
        self.qualification = qualification or getattr(operator, "qualification_report", None)
        self.domain = domain
        self.allow_fixture = allow_fixture
        # Unqualified production physics is never silently activated.
        if not require_qualified and not allow_fixture:
            raise ValueError("bypassing qualification is restricted to labelled mechanism fixtures")
        if require_qualified:
            require_qualified_operator(operator, self.qualification, domain=domain, allow_fixture=allow_fixture)

    def train(self, mode=True):
        super().train(mode)
        self.operator.eval()
        return self

    def forward(self, properties, observations, mask, frequencies_ghz, context,
                baseline_properties=None, baseline_valid=None, noise_scale=None,
                covariance=None, observation_domain=None):
        self.operator.eval()
        report = require_qualified_operator(self.operator, self.qualification, context,
                                             self.domain, self.allow_fixture)
        f = torch.as_tensor(frequencies_ghz).detach().cpu().flatten().tolist()
        allowed = report["frequencies_ghz"]
        if any(not any(abs(float(v)-float(a)) < 1e-5 for a in allowed) for v in f):
            raise ValueError("Phi queried outside qualified frequency support")
        if baseline_properties is not None and properties.ndim == 6 and baseline_properties.ndim == 5:
            if not context.get("constant_spectrum", False):
                raise ValueError("dispersive monitoring needs frequency-resolved baseline properties")
            baseline_properties = baseline_properties[:, :, None].expand_as(properties)
        bounds = report.get("property_bounds")
        if bounds is not None:
            for channel in range(2):
                lo, hi = bounds[channel]
                checked_properties = properties if baseline_properties is None else baseline_properties.detach()+properties
                detached = checked_properties[:, channel].detach()
                if (detached < lo).any() or (detached > hi).any():
                    raise ValueError("inverse state outside qualified physical property domain")
                if baseline_properties is not None:
                    baseline_channel = baseline_properties[:, channel].detach()
                    if (baseline_channel < lo).any() or (baseline_channel > hi).any():
                        raise ValueError("baseline state outside qualified physical property domain")
        if baseline_properties is None:
            if normalize_domain(observation_domain) not in (None, self.domain):
                raise ValueError("absolute consistency requires matching original calibrated observations")
            predicted = self.operator(properties, frequencies_ghz, context)
        else:
            if observation_domain != "current_minus_baseline":
                raise ValueError("monitoring requires explicit current_minus_baseline observation domain")
            if baseline_valid is None:
                raise ValueError("monitoring requires measured baseline registration/validity")
            valid = torch.as_tensor(baseline_valid, device=mask.device).bool()
            while valid.ndim < mask.ndim:
                valid = valid.unsqueeze(-1)
            mask = mask.bool() & valid
            base = baseline_properties.detach()
            # properties is signed patient change, not an iteration correction.
            predicted = self.operator(base+properties, frequencies_ghz, context)-self.operator(base, frequencies_ghz, context)
        return masked_complex_loss(predicted, observations, mask, noise_scale, covariance)


def physics_status(requested="A", operator=None, report=None, required=False):
    if requested not in ("A", "B", "C"):
        raise ValueError("physics mode must be A, B, or C")
    active = requested == "A"
    reason = "supervised_no_phi"
    if requested != "A":
        try:
            if operator is None:
                raise ValueError("no qualified Phi supplied")
            require_qualified_operator(operator, report, domain=operator.domain)
            active, reason = True, "qualified_operator"
        except ValueError as exc:
            reason = str(exc)
            if required:
                raise ValueError(f"physics {requested} explicitly required: {reason}") from exc
    return {"physics_requested": requested, "physics_active": requested if active else "A",
            "physics_reason": reason, "inference_requires_phi": requested == "C" and active}


def forward_training_loss(operator, properties, observations, mask, frequencies_ghz,
                          context, baseline_properties=None, baseline_observations=None,
                          baseline_mask=None, differential_weight=1.0, noise_scale=None):
    """Independent Phi objective with paired differential supervision when exported."""
    prediction = operator(properties, frequencies_ghz, context)
    full = masked_complex_loss(prediction, observations, mask, noise_scale)
    differential = full*0.0
    if baseline_properties is not None:
        if baseline_observations is None or baseline_mask is None:
            raise ValueError("paired differential training needs baseline data and its mask")
        predicted_base = operator(baseline_properties.detach(), frequencies_ghz, context)
        support = mask.bool() & baseline_mask.bool()
        differential = masked_complex_loss(prediction-predicted_base,
            observations-baseline_observations, support, noise_scale)
    return {"total": full+differential_weight*differential,
            "full_signal": full, "differential_signal": differential}


def field_equation_loss(fields, sources, operator, boundary_support, *, context,
                        valid_support=None):
    """Actual supplied discrete field-equation residual; disabled without prerequisites.

    ``operator(fields, context)`` applies an independently defined Maxwell/field
    discretization including material and excitation conventions. Supplying a
    property-image gradient here is a contract violation, not a Maxwell loss.
    Boundary support and source tensors must share field axes; all boundary
    terms are part of the supplied operator/context rather than invented here.
    """
    if fields is None or sources is None or operator is None or boundary_support is None:
        raise ValueError("field loss requires actual fields, sources, operator, and boundary support")
    required = {"equation_name", "material_convention", "boundary_condition", "operator_validated"}
    if not required.issubset(context) or not context["operator_validated"]:
        raise ValueError("field discretization conventions and independent validation required")
    if fields.shape != sources.shape or boundary_support.shape != fields.shape:
        raise ValueError("field/source/boundary axes must match")
    residual = operator(fields, context)-sources
    support = torch.ones_like(residual, dtype=torch.bool) if valid_support is None else valid_support.bool()
    selected = residual[support]
    if not selected.numel():
        return residual.real.sum()*0
    return selected.abs().square().mean()


def validation_signal_metrics(predicted, observed, mask, *, baseline_predicted=None,
                              baseline_observed=None, baseline_mask=None,
                              noise_scale=None, roi_channel_mask=None, floor=1e-8):
    """Held-out qualification metrics; unavailable evidence remains NaN and fails.

    ROI support must be supplied by the data's verified channel/ROI sensitivity
    mapping. It is not inferred from the current lesion label or signal strength.
    Return scalar dictionary suitable for ``qualification_report``. For cohorts,
    concatenate compatible batches before calling; retain per-group reports too.
    """
    def relative(p, y, m):
        mse = masked_complex_loss(p, y, m)
        energy = masked_complex_loss(torch.zeros_like(y), y, m)
        return float((mse/(energy+floor)).sqrt().detach())
    full = relative(predicted, observed, mask)
    support = torch.broadcast_to(mask.bool(), predicted.shape)
    p = torch.where(support, predicted, torch.zeros_like(predicted))
    y = torch.where(support, observed, torch.zeros_like(observed))
    # Worst physical Rx/Tx path, retaining all valid frequencies and batches.
    error = ((p-y).square()).sum((0, 1, 2))
    energy = y.square().sum((0, 1, 2))
    valid_paths = support.any((0, 1, 2))
    worst = float((error[valid_paths]/(energy[valid_paths]+floor)).sqrt().max()) if valid_paths.any() else float("nan")
    diff = float("nan")
    if baseline_predicted is not None and baseline_observed is not None and baseline_mask is not None:
        diff = relative(predicted-baseline_predicted, observed-baseline_observed,
                        mask.bool() & baseline_mask.bool())
    noise = float(masked_complex_loss(predicted, observed, mask, noise_scale).sqrt()) if noise_scale is not None else float("nan")
    roi = relative(predicted, observed, mask.bool() & roi_channel_mask.bool()) if roi_channel_mask is not None else float("nan")
    return {"full_relative_rmse": full, "differential_relative_rmse": diff,
            "noise_relative_rmse": noise, "worst_channel_relative_rmse": worst,
            "roi_relative_rmse": roi}
