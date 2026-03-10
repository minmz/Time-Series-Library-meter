import torch


# 新增状态特征模块

STATE_FEATURE_DIM = 11


def _safe_std(x, dims):
    return torch.std(x, dim=dims, unbiased=False)


def _autocorr(signal, lag):
    if signal.size(1) <= lag:
        return torch.zeros(signal.size(0), device=signal.device, dtype=signal.dtype)

    lhs = signal[:, lag:]
    rhs = signal[:, :-lag]
    lhs = lhs - lhs.mean(dim=1, keepdim=True)
    rhs = rhs - rhs.mean(dim=1, keepdim=True)

    numerator = (lhs * rhs).mean(dim=1)
    denominator = torch.sqrt(lhs.pow(2).mean(dim=1) * rhs.pow(2).mean(dim=1)).clamp_min(1e-6)
    return numerator / denominator


def _channel_correlation_mean(x):
    batch, seq_len, channels = x.shape
    if channels <= 1:
        return torch.zeros(batch, device=x.device, dtype=x.dtype)

    centered = x - x.mean(dim=1, keepdim=True)
    centered = centered.transpose(1, 2)  # [B, C, L]
    cov = torch.bmm(centered, centered.transpose(1, 2)) / max(seq_len - 1, 1)
    var = cov.diagonal(dim1=1, dim2=2).clamp_min(1e-6)
    denom = torch.sqrt(var.unsqueeze(-1) * var.unsqueeze(1))
    corr = cov / denom

    mask = ~torch.eye(channels, dtype=torch.bool, device=x.device)
    masked = corr[:, mask].reshape(batch, -1).abs()
    return masked.mean(dim=1)


def extract_state_features(batch_x):
    if batch_x.ndim != 3:
        raise ValueError("batch_x must have shape [B, L, C].")

    x = batch_x.float()
    batch, seq_len, _ = x.shape
    mean_series = x.mean(dim=2)

    global_mean = x.mean(dim=(1, 2))
    global_std = _safe_std(x, dims=(1, 2))
    abs_mean = x.abs().mean(dim=(1, 2))

    if seq_len > 1:
        last_delta = (x[:, -1, :] - x[:, -2, :]).abs().mean(dim=1)
        volatility = _safe_std(x[:, 1:, :] - x[:, :-1, :], dims=(1, 2))
    else:
        last_delta = torch.zeros(batch, device=x.device, dtype=x.dtype)
        volatility = torch.zeros(batch, device=x.device, dtype=x.dtype)

    start_end_gap = (mean_series[:, -1] - mean_series[:, 0]).abs()

    time_index = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    centered_t = time_index - time_index.mean()
    trend_denominator = (centered_t ** 2).sum().clamp_min(1e-6)
    centered_series = mean_series - mean_series.mean(dim=1, keepdim=True)
    trend = (centered_series * centered_t.unsqueeze(0)).sum(dim=1) / trend_denominator

    lag1 = _autocorr(mean_series, lag=1)
    seasonal_lag = max(2, min(seq_len // 4, 24))
    lagk = _autocorr(mean_series, lag=seasonal_lag)

    spectrum = torch.fft.rfft(mean_series, dim=1)
    power = spectrum.abs().pow(2)
    if power.size(1) > 1:
        power = power[:, 1:]
        dominant_ratio = power.max(dim=1).values / power.sum(dim=1).clamp_min(1e-6)
    else:
        dominant_ratio = torch.zeros(batch, device=x.device, dtype=x.dtype)

    channel_corr = _channel_correlation_mean(x)

    features = torch.stack(
        [
            global_mean,
            global_std,
            abs_mean,
            last_delta,
            start_end_gap,
            trend,
            volatility,
            lag1,
            lagk,
            dominant_ratio,
            channel_corr,
        ],
        dim=1,
    )
    return torch.nan_to_num(features)


def compute_prediction_disagreement(candidate_preds):
    if candidate_preds.ndim != 4:
        raise ValueError("candidate_preds must have shape [B, M, L, C].")

    disagreement = candidate_preds.float().std(dim=1, unbiased=False).mean(dim=(1, 2))
    return torch.nan_to_num(disagreement)
