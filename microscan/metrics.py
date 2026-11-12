"""Physical-unit evaluation; background occupancy never substitutes for skill."""
from __future__ import annotations

import numpy as np
import torch


def property_metrics(predicted, target, foreground=None, scales=(44., .8)):
    """Both physical properties and equally weighted foreground/background error.

    No clinical localization is inferred without a supplied foreground label.
    Empty foreground cases retain their whole-background false-positive error.
    """
    p = torch.as_tensor(predicted).detach().double()
    t = torch.as_tensor(target, device=p.device).double()
    if p.shape != t.shape or p.ndim != 5 or p.shape[1] != 2:
        raise ValueError("Properties must have matching B,2,Z,Y,X shapes")
    if not torch.isfinite(p).all() or not torch.isfinite(t).all():
        raise ValueError("Nonfinite property metric inputs")
    scale = p.new_tensor(scales).reshape(1, 2, 1, 1, 1)
    error = p-t
    result = {}
    for c, name in enumerate(("epsilon_r", "conductivity_s_per_m")):
        result[name+"_mae"] = error[:, c].abs().mean().item()
        result[name+"_rmse"] = error[:, c].square().mean().sqrt().item()
    result["normalized_mse"] = (error/scale).square().mean().item()
    if foreground is not None:
        fg = torch.broadcast_to(torch.as_tensor(foreground, device=p.device).bool(), p.shape)
        normalized = (error/scale).square()
        terms = [normalized[m].mean() for m in (fg, ~fg) if m.any()]
        result["balanced_contrast_mse"] = torch.stack(terms).mean().item()
        result["foreground_normalized_mse"] = normalized[fg].mean().item() if fg.any() else None
        result["background_normalized_mse"] = normalized[~fg].mean().item() if (~fg).any() else None
    return result


def lesion_components(probabilities, grid, schema="multiclass", threshold=.5,
                      class_names=None, roi=None, roi_valid=False):
    """Connected-component reports, with explicit ZYX arrays versus XYZ mm.

    grid requires x_mm/y_mm/z_mm cell centres (or affine_zyx_to_xyz_mm).
    This reports model probabilities; it does not establish calibration.
    """
    from scipy.ndimage import label
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 4 or not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite [class,Z,Y,X]")
    if schema not in ("binary", "multiclass", "multilabel"):
        raise ValueError("Unknown segmentation schema")
    axes = [np.asarray(grid[k], dtype=float) for k in ("z_mm", "y_mm", "x_mm")]
    if tuple(len(a) for a in axes) != p.shape[1:]:
        raise ValueError("Physical grid does not match segmentation axes")
    if any(len(a) < 2 or not np.allclose(np.diff(a), np.diff(a)[0]) for a in axes):
        raise ValueError("Component volume needs uniform cell-centred axes")
    voxel_mm3 = float(abs(np.prod([a[1]-a[0] for a in axes])))
    support = np.ones(p.shape[1:], dtype=bool) if roi is None else np.asarray(roi, bool).reshape(p.shape[1:])
    prediction = p.argmax(0) if schema == "multiclass" else None
    results = []
    classes = range(1, len(p)) if schema == "multiclass" else range(len(p))
    for c in classes:
        mask = ((prediction == c) if prediction is not None else (p[c] >= threshold)) & support
        components, count = label(mask)
        for component in range(1, count+1):
            selected = components == component
            iz, iy, ix = np.nonzero(selected)
            centre = [float(axes[2][ix].mean()), float(axes[1][iy].mean()), float(axes[0][iz].mean())]
            volume = float(selected.sum()*voxel_mm3)
            results.append({"class_id": c, "class_name": class_names[c] if class_names else str(c),
                            "centre_xyz_mm": centre, "volume_mm3": volume, "volume_ml": volume/1000.,
                            "mean_probability": float(p[c][selected].mean()), "voxel_count": int(selected.sum()),
                            "roi_assessed": bool(roi_valid), "probability_calibrated": False})
    return {"components": results, "array_axes": "class,Z,Y,X", "coordinates": "XYZ_mm",
            "roi_valid": bool(roi_valid), "outside_roi_assessed": roi is None,
            "probability_calibrated": False}


def component_changes(current, baseline, baseline_valid):
    """Signed class-volume change; spatial component matching is not assumed."""
    if not baseline_valid:
        return {"valid": False, "reason": "missing_or_unregistered_baseline"}
    def volumes(report):
        out = {}
        for component in report["components"]:
            c = str(component["class_id"])
            out[c] = out.get(c, 0.)+component["volume_ml"]
        return out
    now, before = volumes(current), volumes(baseline)
    return {"valid": True, "class_volume_change_ml": {c: now.get(c, 0.)-before.get(c, 0.) for c in now.keys() | before.keys()}}
