import torch
import torch.nn.functional as F

# 新增案例库模块


class CaseBank:
    def __init__(self, topn=10):
        self.topn = topn
        self.features = None
        self.norm_features = None
        self.best_model = None
        self.sample_errors = None
        self.candidate_count = 0

    def fit(self, features, best_model, sample_errors):
        self.features = features.float().cpu()
        self.norm_features = F.normalize(self.features, dim=1)
        self.best_model = best_model.long().cpu()
        self.sample_errors = sample_errors.float().cpu()
        self.candidate_count = int(self.sample_errors.shape[1])

    def query(self, query_features):
        if self.features is None or self.features.numel() == 0:
            batch = int(query_features.shape[0])
            zeros = torch.zeros(batch, self.candidate_count)
            return {"model_prior": zeros, "error_prior": zeros}

        query = query_features.float().cpu()
        query = F.normalize(query, dim=1)
        similarities = torch.matmul(query, self.norm_features.t())

        topn = min(self.topn, self.features.shape[0])
        indices = similarities.topk(topn, dim=1).indices

        gathered_best = self.best_model[indices]
        gathered_errors = self.sample_errors[indices]

        model_prior = F.one_hot(gathered_best, num_classes=self.candidate_count).float().mean(dim=1)
        error_prior = gathered_errors.mean(dim=1)
        return {"model_prior": model_prior, "error_prior": error_prior}
