import numpy as np
import torch
from microscan.metrics import property_metrics, lesion_components, component_changes


def test_balanced_metrics_penalize_small_foreground():
    t = torch.zeros(1, 2, 4, 4, 4)
    p = t.clone(); p[:, :, 0, 0, 0] = torch.tensor([44., .8])
    fg = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool); fg[..., 0, 0, 0] = True
    m = property_metrics(p, t, fg)
    assert m['balanced_contrast_mse'] > m['normalized_mse']*20


def test_components_xyz_units_and_signed_shrinkage():
    grid = dict(x_mm=np.arange(4)*2., y_mm=np.arange(3)*3., z_mm=np.arange(2)*4.)
    p = np.zeros((1, 2, 3, 4)); p[0, 1, 2, 3] = .9
    report = lesion_components(p, grid, 'binary')
    c = report['components'][0]
    assert c['centre_xyz_mm'] == [6., 6., 4.]
    assert c['volume_mm3'] == 24.
    assert component_changes({'components': []}, report, True)['class_volume_change_ml']['0'] == -.024
    assert not component_changes(report, report, False)['valid']
