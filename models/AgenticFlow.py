import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.state_features import STATE_FEATURE_DIM


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.candidate_count = int(getattr(configs, "agentic_candidate_count", 5))
        self.top_k = min(int(getattr(configs, "top_k_candidates", 3)), self.candidate_count)
        hidden = int(getattr(configs, "meta_hidden", 128))
        dropout = float(getattr(configs, "dropout", 0.1))

        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        context_dim = hidden + 1 + self.candidate_count * 3

        self.router_head = nn.Linear(hidden, self.candidate_count)
        self.risk_head = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.candidate_count),
        )

    def _build_context(self, state_embed, disagreement, route_probs, model_prior, error_prior):
        if model_prior is None:
            model_prior = torch.zeros_like(route_probs)
        if error_prior is None:
            error_prior = torch.zeros_like(route_probs)

        context = torch.cat(
            [
                state_embed,
                disagreement.unsqueeze(-1),
                route_probs,
                model_prior,
                error_prior,
            ],
            dim=-1,
        )
        return context

    def forward(self, state_features, candidate_preds, disagreement, model_prior=None, error_prior=None):
        # Route first, then decide whether the initial route is reliable enough.
        state_embed = self.state_encoder(state_features)
        route_logits = self.router_head(state_embed)
        route_probs = torch.softmax(route_logits, dim=-1)

        context = self._build_context(state_embed, disagreement, route_probs, model_prior, error_prior)
        risk_logits = self.risk_head(context).squeeze(-1)
        risk_scores = torch.sigmoid(risk_logits)

        fusion_logits = self.fusion_head(context)

        # Only the router-selected top-k models can participate in revision.
        topk_indices = route_probs.topk(self.top_k, dim=-1).indices
        topk_mask = torch.zeros_like(route_probs).scatter_(1, topk_indices, 1.0)

        masked_route = route_probs * topk_mask
        masked_route = masked_route / masked_route.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        masked_fusion_logits = fusion_logits.masked_fill(topk_mask == 0, -1e9)
        fusion_probs = torch.softmax(masked_fusion_logits, dim=-1)

        initial_pred = torch.sum(
            candidate_preds * masked_route.unsqueeze(-1).unsqueeze(-1), dim=1
        )
        revised_pred = torch.sum(
            candidate_preds * fusion_probs.unsqueeze(-1).unsqueeze(-1), dim=1
        )
        final_weights = (1.0 - risk_scores.unsqueeze(-1)) * masked_route + risk_scores.unsqueeze(-1) * fusion_probs
        final_weights = final_weights / final_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        final_pred = torch.sum(
            candidate_preds * final_weights.unsqueeze(-1).unsqueeze(-1), dim=1
        )

        return {
            "route_logits": route_logits,
            "route_probs": route_probs,
            "risk_logits": risk_logits,
            "risk_scores": risk_scores,
            "fusion_logits": fusion_logits,
            "fusion_probs": fusion_probs,
            "initial_pred": initial_pred,
            "revised_pred": revised_pred,
            "final_pred": final_pred,
            "final_weights": final_weights,
        }
