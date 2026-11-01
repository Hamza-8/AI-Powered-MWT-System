"""Physical regression, stable-only monitoring, and explicit label-schema losses."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def masked_mean(values, valid=None):
    if valid is None:
        return values.mean()
    support = torch.broadcast_to(valid.bool(), values.shape)
    safe = torch.where(support, values, torch.zeros_like(values))
    if not torch.isfinite(safe).all():
        raise ValueError("nonfinite supervised values on valid support")
    return safe.sum()/support.sum().clamp_min(1)


def balanced_property_loss(predicted, target, foreground=None, validity=None,
                           property_scales=(44.0, .8), kind="huber"):
    """Equal aggregate foreground/background weighting; physical channel scales saved.

    On healthy/empty samples only the available background is averaged, avoiding
    a fabricated absent foreground loss. No target-derived field is a model input.
    """
    if predicted.shape != target.shape or predicted.ndim != 5 or predicted.shape[1] != 2:
        raise ValueError("property tensors must match [B,2,Z,Y,X]")
    valid = torch.ones_like(target, dtype=torch.bool) if validity is None else torch.broadcast_to(validity.bool(), target.shape)
    p = torch.where(valid, predicted, torch.zeros_like(predicted))
    y = torch.where(valid, target, torch.zeros_like(target))
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError("nonfinite valid property supervision")
    scale = torch.as_tensor(property_scales, device=p.device, dtype=p.dtype).view(1, 2, 1, 1, 1)
    if (scale <= 0).any():
        raise ValueError("positive fixed physical property scales required")
    if kind == "mse":
        error = ((p-y)/scale).square()
    elif kind == "huber":
        error = F.smooth_l1_loss(p/scale, y/scale, reduction="none", beta=.1)
    else:
        raise ValueError("property loss kind must be mse or huber")
    if foreground is None:
        return masked_mean(error, valid)
    fg = torch.broadcast_to(foreground.bool(), target.shape)
    batch_losses = []
    for i in range(len(error)):
        terms = []
        for region in (fg[i], ~fg[i]):
            support = region & valid[i]
            if support.any():
                terms.append(masked_mean(error[i], support))
        batch_losses.append(sum(terms)/max(len(terms), 1) if terms else error[i].sum()*0)
    return torch.stack(batch_losses).mean()


def gradient_loss(predicted, target, validity=None, property_scales=(44.0, .8)):
    scale = torch.as_tensor(property_scales, device=predicted.device, dtype=predicted.dtype).view(1, 2, 1, 1, 1)
    valid = torch.ones_like(target, dtype=torch.bool) if validity is None else torch.broadcast_to(validity.bool(), target.shape)
    p = torch.where(valid, predicted, torch.zeros_like(predicted))/scale
    y = torch.where(valid, target, torch.zeros_like(target))/scale
    terms = []
    for dim in (2, 3, 4):
        if predicted.shape[dim] < 2:
            continue
        left = [slice(None)]*5
        right = [slice(None)]*5
        left[dim], right[dim] = slice(1, None), slice(None, -1)
        support = valid[tuple(left)] & valid[tuple(right)]
        diff = (p[tuple(left)]-p[tuple(right)])-(y[tuple(left)]-y[tuple(right)])
        terms.append(masked_mean(diff.abs(), support))
    return sum(terms)/max(len(terms), 1) if terms else p.sum()*0


def stable_temporal_loss(change, stable_mask=None, validity=None, property_scales=(44.0, .8)):
    """Only explicitly labelled stable regions are regularized toward no change."""
    if stable_mask is None:
        return change.sum()*0.0
    support = stable_mask.bool()
    if validity is not None:
        support = support & validity.bool()
    return balanced_property_loss(change, torch.zeros_like(change), validity=support,
                                  property_scales=property_scales)


def reconstruction_loss(outputs, target, foreground=None, baseline_target=None,
                        stable_mask=None, physics_loss=None, weights=None,
                        property_scales=(44.0, .8), validity=None,
                        spectral_target=None, spectral_prediction=None,
                        equation_loss=None):
    """Named physical objective. ``coarse``/``refined`` retain direct supervision.

    Monitoring uses explicit ``change`` output and ``target-baseline_target``.
    A temporal term never substitutes for signed longitudinal labels. Physics
    ramping belongs to the stage engine and is reflected in ``weights['data']``.
    """
    w = {"property": 1., "coarse": .5, "change": 1., "data": 1.,
         "gradient": .02, "temporal": .01, "equation": 0., "spectral": .25}
    w.update(weights or {})
    refined = outputs.get("refined", outputs.get("properties"))
    coarse = outputs.get("coarse", refined)
    if refined is None:
        raise ValueError("reconstruction outputs require refined or properties")
    zero = refined.sum()*0.0
    terms = {"property": balanced_property_loss(refined, target, foreground, validity, property_scales),
             "coarse": balanced_property_loss(coarse, target, foreground, validity, property_scales),
             "gradient": gradient_loss(refined, target, validity, property_scales),
             "change": zero, "data": physics_loss if physics_loss is not None else zero,
             "temporal": zero, "equation": equation_loss if equation_loss is not None else zero,
             "spectral": zero}
    if baseline_target is not None:
        if "change" not in outputs:
            raise ValueError("baseline supervision requires separate signed change output")
        terms["change"] = balanced_property_loss(outputs["change"], target-baseline_target,
                                                    foreground, validity, property_scales)
        terms["temporal"] = stable_temporal_loss(outputs["change"], stable_mask, validity, property_scales)
    elif stable_mask is not None:
        raise ValueError("stable monitoring labels require a valid baseline")
    if spectral_target is not None:
        spectral_prediction = spectral_prediction if spectral_prediction is not None else outputs.get("spectral")
        if spectral_prediction is None or spectral_prediction.shape != spectral_target.shape:
            raise ValueError("spectral supervision requires matching B,2,Q,Z,Y,X properties")
        q = spectral_target.shape[2]
        terms["spectral"] = sum(balanced_property_loss(spectral_prediction[:, :, i], spectral_target[:, :, i],
                           foreground, validity, property_scales) for i in range(q))/q
    if w["equation"] and equation_loss is None:
        raise ValueError("nonzero field loss requires actual operator-based residual")
    terms["total"] = sum(w[name]*value for name, value in terms.items())
    return terms


CLINICAL_LABELS = {0: "healthy_background", 1: "ischemia", 2: "hemorrhage", 3: "tumor", 4: "edema"}


def segmentation_loss(logits, labels, validity=None, schema="binary", ignore_index=-1, smooth=1e-6):
    """Dice+BCE/CE with ignored/unavailable labels, including all-healthy images.

    Clinical mutually-exclusive IDs are 0..4. Binary sphere/object labels are
    *not* clinical lesion types. Overlap is allowed only by explicit multilabel.
    """
    if labels is None:
        raise ValueError("missing segmentation labels cannot become background")
    if schema == "multiclass":
        if logits.shape[1] != len(CLINICAL_LABELS):
            raise ValueError("clinical multiclass requires five healthy/lesion channels")
        if labels.ndim == logits.ndim:
            if labels.shape[1] != 1:
                raise ValueError("mutually-exclusive labels require integer class indices")
            labels = labels[:, 0]
        support = labels != ignore_index
        if validity is not None:
            v = validity[:, 0] if validity.ndim == logits.ndim else validity
            support &= v.bool()
        if (labels[support] < 0).any() or (labels[support] >= logits.shape[1]).any():
            raise ValueError("clinical label outside declared schema")
        safe_labels = torch.where(support, labels.long(), torch.zeros_like(labels).long())
        # Index only valid samples before CE; hidden NaN logits must be isolated.
        safe_logits = torch.where(support[:, None], logits, torch.zeros_like(logits))
        ce = masked_mean(F.cross_entropy(safe_logits, safe_labels, reduction="none"), support)
        probs = safe_logits.softmax(1)
        truth = F.one_hot(safe_labels, num_classes=logits.shape[1]).movedim(-1, 1).to(probs.dtype)
        v = support[:, None].expand_as(probs)
    elif schema in ("binary", "multilabel"):
        if schema == "binary" and logits.shape[1] != 1:
            raise ValueError("binary segmentation requires one logit channel")
        if labels.ndim == logits.ndim-1:
            labels = labels[:, None]
        if labels.shape != logits.shape:
            raise ValueError("binary/multilabel labels must match logit axes")
        v = labels != ignore_index
        if validity is not None:
            v &= torch.broadcast_to(validity.bool(), labels.shape)
        truth = torch.where(v, labels, torch.zeros_like(labels)).to(logits.dtype)
        if ((truth[v] < 0) | (truth[v] > 1)).any():
            raise ValueError("binary/multilabel targets must lie in [0,1]")
        safe_logits = torch.where(v, logits, torch.zeros_like(logits))
        ce = masked_mean(F.binary_cross_entropy_with_logits(safe_logits, truth, reduction="none"), v)
        probs = safe_logits.sigmoid()
    else:
        raise ValueError("label schema must be binary, multiclass, or multilabel")
    # Include absent classes: predicts false positives -> nonzero Dice penalty.
    axes = tuple(range(2, logits.ndim))
    p = torch.where(v, probs, torch.zeros_like(probs))
    y = torch.where(v, truth, torch.zeros_like(truth))
    dice = 1-(2*(p*y).sum(axes)+smooth)/(p.sum(axes)+y.sum(axes)+smooth)
    available = v.sum(axes) > 0
    return ce+masked_mean(dice, available)


def repeated_acquisition_loss(prediction, independent_repeat, mask, *,
                              conditionally_unbiased=False, shared_drift=False):
    """Noise2Noise mechanism only with explicit acquisition-noise assumptions."""
    if not conditionally_unbiased or shared_drift:
        raise ValueError("repeated-acquisition targets require unbiased independent conditional noise")
    from .physics import masked_complex_loss
    return masked_complex_loss(prediction, independent_repeat, mask)
