import copy
import json
import os

import torch

# 新增候选模型池模块

def load_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Candidate manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if "candidates" not in manifest or not manifest["candidates"]:
        raise ValueError("Candidate manifest must define a non-empty 'candidates' list.")
    return manifest


def _strip_module_prefix(state_dict):
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


class CandidatePool:
    def __init__(self, model_dict, base_args, device, manifest_path):
        self.model_dict = model_dict
        self.base_args = base_args
        self.device = device
        self.manifest_path = manifest_path
        self.manifest = load_manifest(manifest_path)
        self.models = []
        self.specs = []
        self.names = []
        self._load_all()

    @property
    def candidate_count(self):
        return len(self.models)

    def _resolve_path(self, raw_path):
        if os.path.isabs(raw_path):
            return raw_path

        manifest_dir = os.path.dirname(os.path.abspath(self.manifest_path))
        from_manifest = os.path.abspath(os.path.join(manifest_dir, raw_path))
        if os.path.exists(from_manifest):
            return from_manifest
        return os.path.abspath(raw_path)

    def _build_candidate_args(self, spec):
        candidate_args = copy.deepcopy(self.base_args)
        candidate_args.task_name = "long_term_forecast"
        candidate_args.model = spec["name"]

        # Each candidate can keep its own hyper-parameter overrides.
        for key, value in spec.get("overrides", {}).items():
            setattr(candidate_args, key, value)

        return candidate_args

    def _load_single_model(self, spec):
        candidate_args = self._build_candidate_args(spec)
        model_cls = self.model_dict[spec["name"]]
        ctor_kwargs = spec.get("constructor_kwargs", {})
        model = model_cls(candidate_args, **ctor_kwargs).float().to(self.device)

        checkpoint_path = self._resolve_path(spec["checkpoint"])
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint for candidate '{spec['name']}' was not found: {checkpoint_path}"
            )

        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]

        state = _strip_module_prefix(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[CandidatePool] '{spec['name']}' loaded with "
                f"{len(missing)} missing and {len(unexpected)} unexpected keys."
            )

        model.eval()
        return model

    def _load_all(self):
        for spec in self.manifest["candidates"]:
            if "name" not in spec or "checkpoint" not in spec:
                raise ValueError("Each candidate must define 'name' and 'checkpoint'.")

            self.models.append(self._load_single_model(spec))
            self.specs.append(spec)
            self.names.append(spec["name"])

    def predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # Reuse the standard decoder-input construction so every candidate
        # follows the same forecasting interface as the original experiments.
        dec_inp = torch.zeros_like(batch_y[:, -self.base_args.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : self.base_args.label_len, :], dec_inp], dim=1).to(self.device)

        predictions = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                predictions.append(outputs[:, -self.base_args.pred_len :, :].detach())

        return torch.stack(predictions, dim=1)
