import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.candidate_pool import CandidatePool, load_manifest
from utils.case_bank import CaseBank
from utils.meta_cache import MetaCacheDataset, MetaCacheWriter, load_meta_cache
from utils.metrics import metric
from utils.state_features import compute_prediction_disagreement, extract_state_features

# 智能体：核心入口：显式workflow
# 流程固定为：状态提取 -> 候选模型预测 -> 路由 -> 风险判断 -> 修正融合

class Exp_Agentic_Flow_Forecast(Exp_Basic):
    def __init__(self, args):
        self.manifest = load_manifest(args.candidate_manifest)
        args.agentic_candidate_count = len(self.manifest["candidates"])
        super().__init__(args)
        self.candidate_pool = CandidatePool(
            self.model_dict, self.args, self.device, self.args.candidate_manifest
        )
        self.case_bank = None

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _build_nonshuffle_loader(self, flag):
        dataset, _ = self._get_data(flag)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
        )
        return dataset, loader

    def _cache_tag(self):
        return self.manifest.get(
            "tag", os.path.splitext(os.path.basename(self.args.candidate_manifest))[0]
        )

    def _cache_path(self, flag):
        filename = (
            f"{self._cache_tag()}_{self.args.data}_sl{self.args.seq_len}"
            f"_pl{self.args.pred_len}_{flag}.pt"
        )
        return os.path.join(self.args.meta_cache_dir, filename)

    def _meta_checkpoint_path(self, setting):
        return os.path.join(self.args.checkpoints, setting, "agentic_flow_meta.pth")

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.meta_lr)

    def _samplewise_candidate_errors(self, candidate_preds, true):
        return torch.mean((candidate_preds - true.unsqueeze(1)) ** 2, dim=(2, 3))

    def _prepare_meta_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        candidate_preds = self.candidate_pool.predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
        f_dim = -1 if self.args.features == "MS" else 0
        candidate_preds = candidate_preds[:, :, :, f_dim:]
        true = batch_y[:, -self.args.pred_len :, f_dim:]

        sample_errors = self._samplewise_candidate_errors(candidate_preds, true)
        best_model = sample_errors.argmin(dim=1)
        disagreement = compute_prediction_disagreement(candidate_preds)

        # If oracle fusion beats the best single candidate by a margin,
        # we treat the sample as revision-worthy during meta training.
        oracle_weights = torch.softmax(-sample_errors, dim=1)
        oracle_pred = torch.sum(
            candidate_preds * oracle_weights.unsqueeze(-1).unsqueeze(-1), dim=1
        )
        oracle_error = torch.mean((oracle_pred - true) ** 2, dim=(1, 2))
        best_single_error = sample_errors.min(dim=1).values
        risk_label = (
            oracle_error + self.args.revision_margin < best_single_error
        ).float()

        state_features = extract_state_features(batch_x)
        return {
            "state_features": state_features,
            "candidate_preds": candidate_preds,
            "true": true,
            "sample_errors": sample_errors,
            "best_model": best_model,
            "disagreement": disagreement,
            "risk_label": risk_label,
        }

    def _collect_meta_cache(self, flag):
        cache_path = self._cache_path(flag)
        if os.path.exists(cache_path) and not self.args.overwrite_meta_cache:
            print(f"[AgenticFlow] Reusing cached meta file: {cache_path}")
            return load_meta_cache(cache_path)

        _, loader = self._build_nonshuffle_loader(flag)
        writer = MetaCacheWriter()
        metadata = {"candidate_names": self.candidate_pool.names, "split": flag}

        self.model.eval()
        for batch in loader:
            # Each raw forecasting window is converted into a lightweight
            # meta sample so the policy network can be trained offline.
            meta = self._prepare_meta_batch(batch)
            writer.append(
                meta["state_features"],
                meta["candidate_preds"],
                meta["true"],
                meta["sample_errors"],
                meta["best_model"],
                meta["disagreement"],
                meta["risk_label"],
            )

        print(f"[AgenticFlow] Meta cache saved to {cache_path}")
        return writer.finalize(cache_path, metadata=metadata)

    def _ensure_meta_cache(self, flag):
        cache_path = self._cache_path(flag)
        if os.path.exists(cache_path) and not self.args.overwrite_meta_cache:
            return load_meta_cache(cache_path)
        return self._collect_meta_cache(flag)

    def _build_case_bank(self, train_cache):
        bank = CaseBank(topn=self.args.case_topn)
        bank.fit(
            train_cache["state_features"],
            train_cache["best_model"],
            train_cache["sample_errors"],
        )
        return bank

    def _query_case_bank(self, state_features):
        if self.case_bank is None:
            zeros = torch.zeros(
                state_features.size(0),
                self.args.agentic_candidate_count,
                device=self.device,
            )
            return zeros, zeros

        priors = self.case_bank.query(state_features.detach().cpu())
        model_prior = priors["model_prior"].to(self.device)
        error_prior = priors["error_prior"].to(self.device)
        return model_prior, error_prior

    def _run_meta_model(self, batch):
        state_features = batch["state_features"].float().to(self.device)
        candidate_preds = batch["candidate_preds"].float().to(self.device)
        true = batch["true"].float().to(self.device)
        disagreement = batch["disagreement"].float().to(self.device)

        model_prior, error_prior = self._query_case_bank(state_features)
        outputs = self.model(
            state_features=state_features,
            candidate_preds=candidate_preds,
            disagreement=disagreement,
            model_prior=model_prior,
            error_prior=error_prior,
        )

        threshold = self.args.risk_threshold
        # Use the initial route unless the predicted risk is high enough
        # to justify the more expensive revision branch.
        revise_mask = outputs["risk_scores"] >= threshold
        threshold_pred = torch.where(
            revise_mask.view(-1, 1, 1),
            outputs["revised_pred"],
            outputs["initial_pred"],
        )
        outputs["threshold_pred"] = threshold_pred
        outputs["true"] = true
        return outputs

    def _evaluate_meta_loader(self, loader):
        self.model.eval()
        total_loss = []

        preds = []
        trues = []
        weights = []
        risk_scores = []

        with torch.no_grad():
            for batch in loader:
                outputs = self._run_meta_model(batch)
                loss = F.mse_loss(outputs["threshold_pred"], outputs["true"])
                total_loss.append(loss.item())

                preds.append(outputs["threshold_pred"].detach().cpu())
                trues.append(outputs["true"].detach().cpu())
                weights.append(outputs["final_weights"].detach().cpu())
                risk_scores.append(outputs["risk_scores"].detach().cpu())

        average_loss = float(np.mean(total_loss)) if total_loss else 0.0
        return {
            "loss": average_loss,
            "preds": torch.cat(preds, dim=0) if preds else torch.empty(0),
            "trues": torch.cat(trues, dim=0) if trues else torch.empty(0),
            "weights": torch.cat(weights, dim=0) if weights else torch.empty(0),
            "risk_scores": torch.cat(risk_scores, dim=0) if risk_scores else torch.empty(0),
        }

    def _train_meta_policy(self, setting):
        train_cache = self._ensure_meta_cache("train")
        val_cache = self._ensure_meta_cache("val")

        self.case_bank = self._build_case_bank(train_cache)

        train_loader = DataLoader(
            MetaCacheDataset(train_cache),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        val_loader = DataLoader(
            MetaCacheDataset(val_cache),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        optimizer = self._select_optimizer()
        checkpoint_path = self._meta_checkpoint_path(setting)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        best_val = float("inf")
        best_state = None

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []

            for batch in train_loader:
                optimizer.zero_grad()

                outputs = self._run_meta_model(batch)
                best_model = batch["best_model"].long().to(self.device)
                risk_label = batch["risk_label"].float().to(self.device)

                route_loss = F.cross_entropy(outputs["route_logits"], best_model)
                risk_loss = F.binary_cross_entropy_with_logits(outputs["risk_logits"], risk_label)
                fusion_loss = F.mse_loss(outputs["final_pred"], outputs["true"])
                total_loss = (
                    fusion_loss
                    + self.args.route_loss_weight * route_loss
                    + self.args.risk_loss_weight * risk_loss
                )

                total_loss.backward()
                optimizer.step()
                train_losses.append(total_loss.item())

            validation = self._evaluate_meta_loader(val_loader)
            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            print(
                f"[AgenticFlow] Epoch {epoch + 1}/{self.args.train_epochs} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {validation['loss']:.6f}"
            )

            if validation["loss"] < best_val:
                best_val = validation["loss"]
                best_state = copy.deepcopy(self._unwrap_model().state_dict())

        if best_state is None:
            raise RuntimeError("Meta-policy training did not produce a valid checkpoint.")

        torch.save(best_state, checkpoint_path)
        self._unwrap_model().load_state_dict(best_state)
        print(f"[AgenticFlow] Best meta-policy checkpoint saved to {checkpoint_path}")

    def train(self, setting):
        if self.args.agentic_stage in ["collect_meta", "full"]:
            for flag in ["train", "val", "test"]:
                self._collect_meta_cache(flag)

        if self.args.agentic_stage in ["train_meta", "full"]:
            self._train_meta_policy(setting)

        return self.model

    def _load_meta_policy(self, setting):
        checkpoint_path = self._meta_checkpoint_path(setting)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Meta-policy checkpoint not found: {checkpoint_path}. "
                "Run train_meta or full stage first."
            )

        state = torch.load(checkpoint_path, map_location=self.device)
        self._unwrap_model().load_state_dict(state)
        self.model.eval()

    def test(self, setting, test=0):
        if self.args.agentic_stage == "collect_meta" and self.args.is_training:
            print("[AgenticFlow] collect_meta stage finished. Skip evaluation.")
            return

        self._load_meta_policy(setting)

        train_cache = self._ensure_meta_cache("train")
        self.case_bank = self._build_case_bank(train_cache)
        test_cache = self._ensure_meta_cache("test")

        test_loader = DataLoader(
            MetaCacheDataset(test_cache),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        evaluation = self._evaluate_meta_loader(test_loader)

        preds = evaluation["preds"].numpy()
        trues = evaluation["trues"].numpy()
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"[AgenticFlow] mse:{mse}, mae:{mae}")

        folder_path = os.path.join("./results", setting)
        os.makedirs(folder_path, exist_ok=True)

        np.save(os.path.join(folder_path, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, "pred.npy"), preds)
        np.save(os.path.join(folder_path, "true.npy"), trues)
        np.save(os.path.join(folder_path, "weights.npy"), evaluation["weights"].numpy())
        np.save(os.path.join(folder_path, "risk_scores.npy"), evaluation["risk_scores"].numpy())

        with open("result_agentic_flow_forecast.txt", "a", encoding="utf-8") as handle:
            handle.write(setting + "  \n")
            handle.write(f"mse:{mse}, mae:{mae}\n\n")
