"""MicroScan neural models, integrating the two historical prototype branches.

Conv1D-to-volume inversion derives from FCERNN; the exact adapter and full-3D
residual enhancement are engineering adaptations. There is no recurrent unit.
All public measurements use B,2,F,Rx,Tx; volumes use B,property,Z,Y,X.
"""
from __future__ import annotations
import math
from torch.utils.checkpoint import checkpoint
from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F


def groups(channels):
    return max(g for g in range(1, min(8, channels) + 1) if channels % g == 0 and channels // g >= 2) if channels > 1 else 1


def sanitize_measurements(values, mask):
    if values.ndim != 5 or values.shape[1] != 2:
        raise ValueError("Measurements must be [B,2,F,Rx,Tx]")
    if mask.shape != (values.shape[0], 1, *values.shape[2:]):
        raise ValueError("Observation mask must be [B,1,F,Rx,Tx]")
    support = mask.bool().expand_as(values)
    if not torch.isfinite(values[support]).all():
        raise ValueError("Observed measurements contain nonfinite values")
    return torch.where(support, values, torch.zeros_like(values))


def measurement_features(values, mask, frequencies_ghz):
    values = sanitize_measurements(values, mask)
    freq = torch.as_tensor(frequencies_ghz, device=values.device, dtype=values.dtype)
    if freq.ndim == 1:
        freq = freq[None].expand(values.shape[0], -1)
    if freq.shape != (values.shape[0], values.shape[2]):
        raise ValueError("Frequency coordinate grid must match the fixed slots")
    # Physical frequency and signed port coordinates remain explicit.
    rx = torch.linspace(-1, 1, values.shape[3], device=values.device, dtype=values.dtype)
    tx = torch.linspace(-1, 1, values.shape[4], device=values.device, dtype=values.dtype)
    shape = (values.shape[0], 1, *values.shape[2:])
    fc = freq[:, None, :, None, None].expand(shape)
    rc = rx[None, None, None, :, None].expand(shape)
    tc = tx[None, None, None, None, :].expand(shape)
    return torch.cat((values, mask.to(values), fc, rc, tc), 1)


class ResidualBlock3D(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(incoming, outgoing, 3, padding=1, bias=False),
                                  nn.GroupNorm(groups(outgoing), outgoing), nn.SiLU(),
                                  nn.Conv3d(outgoing, outgoing, 3, padding=1, bias=False),
                                  nn.GroupNorm(groups(outgoing), outgoing))
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv3d(incoming, outgoing, 1)
    def forward(self, x):
        return F.silu(self.body(x) + self.skip(x))


def match_shape(x, target):
    """Symmetric pad/crop only; never rescale microwave measurement axes."""
    slices = [slice(None), slice(None)]
    for size, wanted in zip(x.shape[2:], target):
        start = max(0, (size - wanted) // 2)
        slices.append(slice(start, start + min(size, wanted)))
    x = x[tuple(slices)]
    pad = []
    for size, wanted in reversed(list(zip(x.shape[2:], target))):
        d = wanted - size
        pad.extend((d // 2, d - d // 2))
    return F.pad(x, pad)


class ResidualUNet3D(nn.Module):
    def __init__(self, input_channels, output_channels, widths=(16, 32, 64, 128), zero_output=False):
        super().__init__()
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        previous = input_channels
        for i, width in enumerate(widths):
            self.encoders.append(ResidualBlock3D(previous, width))
            if i < len(widths) - 1:
                self.downs.append(nn.Conv3d(width, widths[i+1], 3, stride=2, padding=1))
                previous = widths[i+1]
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for a, b in zip(reversed(widths[1:]), reversed(widths[:-1])):
            self.ups.append(nn.ConvTranspose3d(a, b, 2, stride=2))
            self.decoders.append(ResidualBlock3D(b * 2, b))
        self.output = nn.Conv3d(widths[0], output_channels, 1)
        if zero_output:
            nn.init.normal_(self.output.weight, std=1e-4)
            nn.init.zeros_(self.output.bias)
        self.bottleneck_channels = widths[-1]
        self.gradient_checkpointing = False
    def forward(self, x, conditioning=None):
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = checkpoint(encoder,x,use_reentrant=False) if self.gradient_checkpointing and self.training else encoder(x)
            if i < len(self.downs):
                skips.append(x)
                x = self.downs[i](x)
        if conditioning is not None:
            x = x + conditioning[:, :, None, None, None]
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = match_shape(up(x), skip.shape[2:])
            decoder_input = torch.cat((x,skip),1)
            x = checkpoint(decoder,decoder_input,use_reentrant=False) if self.gradient_checkpointing and self.training else decoder(decoder_input)
        return self.output(x)


class CoarseInverse(nn.Module):
    """Eight Conv1D stages; explicit pooled adapter; learned native 3D decoder."""
    def __init__(self, config=None):
        super().__init__()
        c = config or {}
        widths = c.get("encoder_channels", (32,32,64,64,128,128,256,256))
        kernels = c.get("encoder_kernels", (7,7,5,5,3,3,3,3))
        strides = c.get("encoder_strides", (2,1,4,1,2,1,4,1))
        if len(widths) != 8:
            raise ValueError("Coarse inverse requires eight encoder stages")
        layers, previous = [], 6
        for width, kernel, stride in zip(widths, kernels, strides):
            layers.extend((nn.Conv1d(previous, width, kernel, stride, kernel//2, bias=False),
                           nn.GroupNorm(groups(width), width), nn.SiLU()))
            previous = width
        self.encoder = nn.Sequential(*layers)
        self.latent_channels, self.latent_size = c.get("latent_channels",64), c.get("latent_size",8)
        self.pooled_length = c.get("pooled_length",32)
        elements = self.latent_channels * self.latent_size**3
        if elements % self.pooled_length:
            raise ValueError("Latent elements must divide pooled_length exactly")
        self.adapter = nn.Conv1d(previous, elements//self.pooled_length, 1)
        decoder = []
        previous = self.latent_channels
        widths = c.get("decoder_channels", (128,64,32,16))
        for width in widths:
            decoder.extend((nn.ConvTranspose3d(previous, width, 4, 2, 1, bias=False),
                            nn.GroupNorm(groups(width), width), nn.SiLU()))
            previous = width
        self.decoder = nn.Sequential(*decoder)
        self.output = nn.Conv3d(previous, 2, 1)
        self.output_size = self.latent_size * 2**len(widths)
        self.feature_channels = previous
    def forward(self, features):
        encoded = self.encoder(features.flatten(2))
        pooled = F.adaptive_avg_pool1d(encoded, self.pooled_length)
        latent = self.adapter(pooled).reshape(features.shape[0], self.latent_channels,
                                              self.latent_size,self.latent_size,self.latent_size)
        decoded = self.decoder(latent)
        return self.output(decoded), decoded


class MeasurementEncoder(nn.Module):
    """Compact mask/frequency/port-aware conditioning, without per-port image volumes."""
    def __init__(self, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv3d(6, 16, 3, padding=1), nn.SiLU(),
                                    nn.Conv3d(16, 32, 3, stride=(1,2,2), padding=1), nn.SiLU())
        self.readout = nn.Linear(32, outgoing)
    def forward(self, features, mask):
        encoded = self.layers(features)
        support = F.adaptive_max_pool3d(mask.to(features), encoded.shape[2:])
        pooled = (encoded * support).sum((2,3,4))/support.sum((2,3,4)).clamp_min(1)
        return self.readout(pooled)


class EnhancementUNet(nn.Module):
    def __init__(self, widths=(16,32,64,128)):
        super().__init__()
        # coarse2 + baseline2 + ROI1 + baseline-valid1 + ROI-valid1 + mode1
        self.unet = ResidualUNet3D(8, 2, widths, zero_output=True)
        self.measurement_encoder = MeasurementEncoder(widths[-1])
        self.step_logit = nn.Parameter(torch.tensor(-2.0))
    def forward(self, coarse, features, mask, baseline=None, roi=None, baseline_valid=None, monitoring=False):
        shape = (coarse.shape[0],1,*coarse.shape[2:])
        if baseline is None:
            baseline = torch.zeros_like(coarse)
            bv = torch.zeros(shape, device=coarse.device, dtype=coarse.dtype)
        else:
            bv = torch.ones(shape,device=coarse.device,dtype=coarse.dtype)
            if baseline_valid is not None:
                bv = bv * baseline_valid.reshape(-1,1,1,1,1).to(coarse)
            baseline = torch.where(bv.bool(), baseline, torch.zeros_like(baseline))
        if roi is None:
            roi = torch.ones(shape,device=coarse.device,dtype=coarse.dtype)
            rv = torch.zeros_like(roi)
        else:
            roi = roi.to(coarse)
            rv = torch.ones_like(roi)
        mode = torch.full_like(roi,float(monitoring))
        context = torch.cat((coarse,baseline,roi,bv,rv,mode),1)
        conditioning = self.measurement_encoder(features,mask)
        return coarse + torch.sigmoid(self.step_logit) * self.unet(context, conditioning)


def tiny_config():
    return dict(encoder_channels=[4]*8, latent_channels=4, latent_size=2, pooled_length=4,
                decoder_channels=[8,4,4], enhancement_channels=[4,8,8], correction_depth=1,
                segmentation_channels=[4,8,8], frequency_grid_ghz=[.6,.65,.7,.75,.8,.85,.9,.95,1.,1.05])


class MSRNN(nn.Module):
    """Coarse inverse + shared full-3D residual enhancement. No recurrence.

    A and B instantiate exactly the same graph. B physics losses live outside
    this deployment module. Only explicit C enables Phi feedback.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = dict(config or {})
        c = self.config
        self.register_buffer("frequency_grid", torch.tensor(c.get("frequency_grid_ghz",[.6+.05*i for i in range(10)])))
        self.reference_frequency_ghz = c.get("reference_frequency_ghz",.85)
        self.coarse = CoarseInverse(c)
        widths = c.get("enhancement_channels", (16,32,64,128))
        self.enhancement = EnhancementUNet(widths)
        self.enhancement.unet.gradient_checkpointing = c.get("gradient_checkpointing",False)
        self.correction_depth = c.get("correction_depth",1)
        if self.correction_depth not in (0,1,2,3):
            raise ValueError("correction_depth must be 0..3")
        # Frequency-conditioned residual in physical property units. Query at
        # reference frequency returns the primary map exactly, by construction.
        self.spectral = nn.Sequential(nn.Conv3d(self.coarse.feature_channels,16,1),nn.SiLU(),nn.Conv3d(16,4,1))
        self.register_buffer("property_scale",torch.tensor(c.get("property_scales",[44.,.8])).reshape(1,2,1,1,1))
        self.register_buffer("property_offset",torch.tensor(c.get("property_offsets",[1.,0.])).reshape(1,2,1,1,1))
    def spectral_properties(self, outputs, query_ghz):
        q = torch.as_tensor(query_ghz,device=outputs["refined"].device,dtype=outputs["refined"].dtype)
        if q.ndim==1: q=q[None].expand(outputs["refined"].shape[0],-1)
        if q.ndim!=2 or q.shape[0]!=outputs["refined"].shape[0] or not torch.isfinite(q).all():
            raise ValueError("Spectral queries must be finite [Q] or [B,Q]")
        delta = (q-self.reference_frequency_ghz)[:,None,:,None,None,None]
        coefficients = self.spectral(outputs["features"])
        slope, curve = coefficients.chunk(2,1)
        return outputs["refined"][:,:,None] + (slope[:,:,None]*delta + curve[:,:,None]*delta.square())*self.property_scale[:,:,None]
    def forward(self, measurements, mask, frequencies_ghz=None, baseline=None, roi=None,
                context_validity=None, mode="snapshot",physics_mode="A",phi=None,physics_context=None,physics_observations=None):
        if physics_mode not in ("A","B","C"):
            raise ValueError("physics_mode must be A, B or C")
        if mode not in ("snapshot","monitoring","full_scan","focused","baseline_tracking"):
            raise ValueError("Unknown acquisition mode")
        monitoring = mode in ("monitoring","baseline_tracking")
        if monitoring and context_validity is not None and "measurement_registration" in context_validity:
            registered=torch.as_tensor(context_validity["measurement_registration"],device=mask.device).bool().expand(measurements.shape[0])
            mask=mask.bool() & registered[:,None,None,None,None]
        if measurements.shape[2] != self.frequency_grid.numel():
            raise ValueError("Use the fixed frequency-slot representation; missing frequencies require masks")
        frequencies = self.frequency_grid if frequencies_ghz is None else frequencies_ghz
        if not torch.allclose(torch.as_tensor(frequencies,device=self.frequency_grid.device,dtype=self.frequency_grid.dtype), self.frequency_grid):
            raise ValueError("Frequency grid differs from checkpoint architecture")
        features = measurement_features(measurements,mask,frequencies)
        normalized, latent_features = self.coarse(features)
        coarse = normalized*self.property_scale if monitoring else F.softplus(normalized)*self.property_scale+self.property_offset
        baseline_valid = None if context_validity is None else context_validity.get("baseline")
        if baseline_valid is not None: baseline_valid=torch.as_tensor(baseline_valid,device=measurements.device).expand(measurements.shape[0])
        refined = coarse
        if physics_mode == "C":
            if phi is None:
                raise ValueError("Experimental C feedback requires an explicitly compatible qualified Phi")
            from .physics import require_qualified_operator,freeze_operator
            report=require_qualified_operator(phi,context=physics_context,domain=phi.domain,allow_fixture=bool((physics_context or {}).get("mechanism_fixture",False)))
            if any(not any(abs(float(value)-float(allowed))<1e-5 for allowed in report["frequencies_ghz"]) for value in torch.as_tensor(frequencies).flatten()):
                raise ValueError("C queried outside qualified frequency support")
            expected_domain="current_minus_baseline" if monitoring else phi.domain
            if (physics_context or {}).get("observation_domain")!=expected_domain:
                raise ValueError("C requires matching original calibrated observation domain")
            freeze_operator(phi)
            if physics_observations is None:
                raise ValueError("C requires original calibrated physics_observations and an explicit observation domain")
            if not (physics_context or {}).get("observation_domain"):
                raise ValueError("C requires explicit observation_domain")
        for _ in range(self.correction_depth):
            conditional_features = features
            if physics_mode == "C":
                state = {"refined":refined,"features":latent_features}
                spectral = self.spectral_properties(state,frequencies)
                bounds=report.get("property_bounds")
                if bounds is not None:
                    checked=spectral if not monitoring else spectral+(physics_context or {}).get("baseline_spectral",torch.zeros_like(spectral))
                    for channel,(lo,hi) in enumerate(bounds):
                        if (checked[:,channel].detach()<lo).any() or (checked[:,channel].detach()>hi).any():
                            raise ValueError("C inverse state outside qualified physical property domain")
                if monitoring:
                    if baseline is None or baseline_valid is None or not baseline_valid.bool().all():
                        raise ValueError("C monitoring requires explicitly valid registered baseline context")
                    baseline_spectral = physics_context.get("baseline_spectral") if physics_context else None
                    if baseline_spectral is None:
                        raise ValueError("C monitoring requires frequency-resolved baseline properties")
                    prediction = phi(baseline_spectral+spectral,frequencies,physics_context)-phi(baseline_spectral,frequencies,physics_context)
                else:
                    prediction = phi(spectral,frequencies,physics_context)
                residual = sanitize_measurements(physics_observations-prediction,mask)
                if physics_context is not None and "measurement_scale" in physics_context:
                    residual=residual/torch.as_tensor(physics_context["measurement_scale"],device=residual.device,dtype=residual.dtype).clamp_min(1e-12)
                conditional_features = measurement_features(residual,mask,frequencies)
            refined = self.enhancement(refined,conditional_features,mask,baseline,roi,baseline_valid,monitoring)
            if not monitoring:
                refined = torch.maximum(refined,self.property_offset)
        valid = mask.bool().flatten(1).any(1)
        if monitoring:
            absolute_valid = valid & (baseline is not None)
            if baseline_valid is not None:
                absolute_valid = absolute_valid & baseline_valid.to(valid.device).bool().reshape(-1)
            change = refined
            absolute = refined + baseline if baseline is not None else torch.full_like(refined,float("nan"))
            absolute = torch.where(absolute_valid[:,None,None,None,None],absolute,torch.full_like(absolute,float("nan")))
        else:
            absolute_valid = valid
            absolute = refined
            change = refined-baseline if baseline is not None else None
        return dict(coarse=coarse,refined=refined,absolute=absolute,change=change,absolute_valid=absolute_valid,
                    measurement_valid=valid,features=latent_features,physics_requested=physics_mode,
                    physics_active="C" if physics_mode=="C" else "A",physics_mechanism_fixture=bool(physics_mode=="C" and (physics_context or {}).get("mechanism_fixture",False)),correction_depth=self.correction_depth,
                    observation_mask=mask,roi=roi,monitoring=monitoring)


CLINICAL_LABELS = {0:"healthy_background",1:"ischemia",2:"hemorrhage",3:"tumor",4:"edema"}


class SegmentationUNet(nn.Module):
    def __init__(self, classes=5, label_mode="multiclass", widths=(16,32,64,128,256)):
        super().__init__()
        if label_mode not in ("binary","multiclass","multilabel"):
            raise ValueError("Invalid segmentation schema")
        if label_mode == "binary": classes=1
        self.label_mode, self.classes = label_mode, classes
        self.unet = ResidualUNet3D(9,classes,widths)
    def forward(self, refined, baseline=None, difference=None, roi=None, detach=True):
        if detach: refined=refined.detach()
        shape=(refined.shape[0],1,*refined.shape[2:])
        baseline_valid = torch.full(shape,float(baseline is not None),device=refined.device,dtype=refined.dtype)
        baseline = torch.zeros_like(refined) if baseline is None else baseline.detach() if detach else baseline
        difference = torch.zeros_like(refined) if difference is None else difference.detach() if detach else difference
        roi_valid = torch.full_like(baseline_valid,float(roi is not None))
        roi = torch.ones_like(baseline_valid) if roi is None else roi.to(refined)
        return self.unet(torch.cat((refined,baseline,difference,roi,baseline_valid,roi_valid),1))
    def probabilities(self, logits):
        return logits.softmax(1) if self.label_mode=="multiclass" else logits.sigmoid()
