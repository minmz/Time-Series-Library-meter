import numpy as np
import torch


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

# 补了样本级误差函数，给路由标签和风险标签用。

# 样本级误差函数
def samplewise_mse(pred, true):
    if isinstance(pred, torch.Tensor):
        reduce_dims = tuple(range(1, pred.ndim))
        return torch.mean((pred - true) ** 2, dim=reduce_dims)
    reduce_dims = tuple(range(1, pred.ndim))
    return np.mean((pred - true) ** 2, axis=reduce_dims)


def samplewise_mae(pred, true):
    if isinstance(pred, torch.Tensor):
        reduce_dims = tuple(range(1, pred.ndim))
        return torch.mean(torch.abs(pred - true), dim=reduce_dims)
    reduce_dims = tuple(range(1, pred.ndim))
    return np.mean(np.abs(pred - true), axis=reduce_dims)
