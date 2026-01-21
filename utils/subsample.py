import torch


def subsample_frequencies(measurement, keep_count):
    num_freq = measurement.shape[0]
    keep_count = min(keep_count, num_freq)
    indices = torch.randperm(num_freq)[:keep_count]
    mask = torch.zeros_like(measurement.real)
    mask[indices] = 1.0
    return measurement * mask, mask


def subsample_antennas(measurement, keep_count):
    tx = measurement.shape[1]
    rx = measurement.shape[2]
    keep_tx = min(keep_count, tx)
    keep_rx = min(keep_count, rx)
    tx_idx = torch.randperm(tx)[:keep_tx]
    rx_idx = torch.randperm(rx)[:keep_rx]
    mask = torch.zeros_like(measurement.real)
    mask[:, tx_idx[:, None], rx_idx] = 1.0
    return measurement * mask, mask


def apply_budget_subsampling(measurement, budget):
    num_freq, tx, rx = measurement.shape
    total = num_freq * tx * rx
    keep = max(1, int(total * budget))
    flat_indices = torch.randperm(total)[:keep]
    mask = torch.zeros(total, device=measurement.device)
    mask[flat_indices] = 1.0
    mask = mask.view(num_freq, tx, rx)
    return measurement * mask, mask
