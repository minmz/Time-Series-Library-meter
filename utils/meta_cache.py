import os

import torch
from torch.utils.data import Dataset

"""
新增决策层训练缓存：先把“每个样本在多个候选模型上的预测表现和决策特征”离线存下来，
供后续路由器、风险模块和融合模块训练使用的缓存数据集。

具体解释：
作用不是缓存原始数据，而是缓存：
1、 当前样本的状态特征
2、 5个候选模型各自的预测结果
3、 5个候选模型各自的误差
4、 哪个模型在这个样本上最好
5、 模型之间的分歧
6、 这个样本是否值得触发修正

也就是说，它缓存的是：
“给上层决策器训练和推理要用的信息”
不是缓存底层时序数据本身。
"""

class MetaCacheWriter:
    def __init__(self):
        self.buffers = {
            "state_features": [],
            "candidate_preds": [],
            "true": [],
            "sample_errors": [],
            "best_model": [],
            "disagreement": [],
            "risk_label": [],
        }

    def append(
        self,
        state_features,
        candidate_preds,
        true,
        sample_errors,
        best_model,
        disagreement,
        risk_label,
    ):
        self.buffers["state_features"].append(state_features.detach().cpu())
        self.buffers["candidate_preds"].append(candidate_preds.detach().cpu())
        self.buffers["true"].append(true.detach().cpu())
        self.buffers["sample_errors"].append(sample_errors.detach().cpu())
        self.buffers["best_model"].append(best_model.detach().cpu())
        self.buffers["disagreement"].append(disagreement.detach().cpu())
        self.buffers["risk_label"].append(risk_label.detach().cpu())

    def finalize(self, save_path, metadata=None):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cache = {}
        for key, values in self.buffers.items():
            cache[key] = torch.cat(values, dim=0) if values else torch.empty(0)
        cache["metadata"] = metadata or {}
        torch.save(cache, save_path)
        return cache


def load_meta_cache(cache_path, map_location="cpu"):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Meta cache not found: {cache_path}")
    return torch.load(cache_path, map_location=map_location)


class MetaCacheDataset(Dataset):
    def __init__(self, cache):
        self.cache = cache
        self.length = int(cache["state_features"].shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            "state_features": self.cache["state_features"][index],
            "candidate_preds": self.cache["candidate_preds"][index],
            "true": self.cache["true"][index],
            "sample_errors": self.cache["sample_errors"][index],
            "best_model": self.cache["best_model"][index],
            "disagreement": self.cache["disagreement"][index],
            "risk_label": self.cache["risk_label"][index],
        }
