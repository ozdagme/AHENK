"""P02-AHENK güvenilirlik kapılı füzyon pilotu.

Metin için train-only TF-IDF+SVD, akustik için otomatik özet özellikleri ve
fiyat için yalnız çağrı öncesi özellikleri kullanır. Tam veri ile sentetik
%25/%50 kip eksikliği ayrı raporlanır. Varsayılan CPU'dur; insan etiketi yoktur.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


EPSILON = 1e-12
TARGET = "target_realized_variance_5d"
MODALITIES = ("text", "acoustic", "price")
MODEL_SPECS = {
    "price_mlp": {"mode": "price", "dropout": 0.0},
    "concat_no_dropout": {"mode": "concat", "dropout": 0.0},
    "concat_modality_dropout": {"mode": "concat", "dropout": 0.25},
    "fixed_mean_dropout": {"mode": "fixed", "dropout": 0.25},
    "ahenk_gate_dropout": {"mode": "gate", "dropout": 0.25},
}


def qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.maximum(np.asarray(actual, dtype=float), EPSILON)
    predicted = np.maximum(np.asarray(predicted, dtype=float), EPSILON)
    ratio = actual / predicted
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def metric_set(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.maximum(np.asarray(actual, dtype=float), EPSILON)
    predicted = np.maximum(np.asarray(predicted, dtype=float), EPSILON)
    return {
        "qlike": qlike(actual, predicted),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
        "log_rmse": float(
            np.sqrt(np.mean(np.square(np.log(actual) - np.log(predicted))))
        ),
    }


def text_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=30000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def read_texts(root: Path, frame: pd.DataFrame) -> list[str]:
    return [
        (root / path).read_text(encoding="utf-8", errors="replace")
        for path in frame["text_path"]
    ]


def fit_dense_text(
    root: Path,
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    vectorizer = text_vectorizer()
    train_sparse = vectorizer.fit_transform(read_texts(root, train))
    dev_sparse = vectorizer.transform(read_texts(root, dev))
    test_sparse = vectorizer.transform(read_texts(root, test))

    components = min(dimensions, train_sparse.shape[0] - 1, train_sparse.shape[1] - 1)
    svd = TruncatedSVD(n_components=components, random_state=20260927)
    train_dense = svd.fit_transform(train_sparse)
    dev_dense = svd.transform(dev_sparse)
    test_dense = svd.transform(test_sparse)
    scaler = StandardScaler()
    train_dense = scaler.fit_transform(train_dense).astype("float32")
    dev_dense = scaler.transform(dev_dense).astype("float32")
    test_dense = scaler.transform(test_dense).astype("float32")
    info = {
        "vocabulary_size": len(vectorizer.vocabulary_),
        "svd_dimensions": components,
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
    }
    return train_dense, dev_dense, test_dense, info


def fit_numeric(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    train_values = imputer.fit_transform(train[columns])
    dev_values = imputer.transform(dev[columns])
    test_values = imputer.transform(test[columns])
    scaler = StandardScaler()
    return (
        scaler.fit_transform(train_values).astype("float32"),
        scaler.transform(dev_values).astype("float32"),
        scaler.transform(test_values).astype("float32"),
    )


def quality_features(
    train: pd.DataFrame,
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text_token_scale = max(float(train["text_token_count"].quantile(0.90)), 1.0)
    text_line_scale = max(float(train["text_nonempty_line_count"].quantile(0.90)), 1.0)
    acoustic_scale = max(float(train["acoustic_segment_count"].quantile(0.90)), 1.0)
    missing_columns = [column for column in train if column.endswith("_missing_rate")]

    outputs = []
    for frame in frames:
        text_q = np.column_stack(
            [
                np.clip(frame["text_token_count"].to_numpy(float) / text_token_scale, 0, 1),
                np.clip(
                    frame["text_nonempty_line_count"].to_numpy(float) / text_line_scale,
                    0,
                    1,
                ),
            ]
        )
        acoustic_missing = frame[missing_columns].mean(axis=1).to_numpy(float)
        acoustic_q = np.column_stack(
            [
                np.clip(1.0 - acoustic_missing, 0, 1),
                np.clip(
                    frame["acoustic_segment_count"].to_numpy(float) / acoustic_scale,
                    0,
                    1,
                ),
            ]
        )
        price_q = np.column_stack(
            [
                np.clip(
                    frame["price_60d_available_returns"].to_numpy(float) / 60.0,
                    0,
                    1,
                ),
                np.clip(frame["price_history_days"].to_numpy(float) / 60.0, 0, 1),
            ]
        )
        outputs.append(np.hstack([text_q, acoustic_q, price_q]).astype("float32"))
    return tuple(outputs)  # type: ignore[return-value]


class FusionNet(nn.Module):
    def __init__(self, dimensions: tuple[int, int, int], mode: str, hidden: int = 64) -> None:
        super().__init__()
        self.mode = mode
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.LayerNorm(hidden))
                for dimension in dimensions
            ]
        )
        if mode == "concat":
            head_input = hidden * 3 + 3
        else:
            head_input = hidden
        if mode == "gate":
            self.gate = nn.Sequential(nn.Linear(9, 32), nn.GELU(), nn.Linear(32, 3))
        else:
            self.gate = None
        self.head = nn.Sequential(
            nn.Linear(head_input, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(
        self,
        text: torch.Tensor,
        acoustic: torch.Tensor,
        price: torch.Tensor,
        quality: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs = (text, acoustic, price)
        representations = torch.stack(
            [encoder(value) for encoder, value in zip(self.encoders, inputs, strict=True)],
            dim=1,
        )
        representations = representations * mask.unsqueeze(-1)

        if self.mode == "price":
            fused = representations[:, 2]
        elif self.mode == "concat":
            fused = torch.cat([representations.flatten(1), mask], dim=1)
        elif self.mode == "fixed":
            denominator = mask.sum(1, keepdim=True).clamp_min(1.0)
            fused = representations.sum(1) / denominator
        elif self.mode == "gate":
            masked_quality = quality * mask.repeat_interleave(2, dim=1)
            logits = self.gate(torch.cat([masked_quality, mask], dim=1))
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=1)
            fused = (representations * weights.unsqueeze(-1)).sum(1)
        else:
            raise ValueError(f"Bilinmeyen mod: {self.mode}")
        return self.head(fused).squeeze(1)


def random_mask(count: int, probability: float, rng: np.random.Generator) -> np.ndarray:
    mask = (rng.random((count, 3)) >= probability).astype("float32")
    empty = np.where(mask.sum(axis=1) == 0)[0]
    if len(empty):
        chosen = rng.integers(0, 3, size=len(empty))
        mask[empty, chosen] = 1.0
    return mask


def tensors(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    quality: np.ndarray,
    target: np.ndarray,
) -> TensorDataset:
    mask = np.ones((len(target), 3), dtype="float32")
    return TensorDataset(
        torch.from_numpy(arrays[0]),
        torch.from_numpy(arrays[1]),
        torch.from_numpy(arrays[2]),
        torch.from_numpy(quality),
        torch.from_numpy(mask),
        torch.from_numpy(np.log(target).astype("float32")),
    )


@torch.no_grad()
def predict(
    model: FusionNet,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    quality: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    lower: float,
    upper: float,
) -> np.ndarray:
    model.eval()
    output = model(
        torch.from_numpy(arrays[0]).to(device),
        torch.from_numpy(arrays[1]).to(device),
        torch.from_numpy(arrays[2]).to(device),
        torch.from_numpy(quality).to(device),
        torch.from_numpy(mask).to(device),
    )
    return np.clip(np.exp(output.cpu().numpy()), lower, upper)


def train_model(
    mode: str,
    dropout_probability: float,
    dimensions: tuple[int, int, int],
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    dev_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    train_quality: np.ndarray,
    dev_quality: np.ndarray,
    train_target: np.ndarray,
    dev_target: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
) -> tuple[FusionNet, dict]:
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = FusionNet(dimensions, mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        tensors(train_arrays, train_quality, train_target),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    lower = float(max(np.quantile(train_target, 0.001), EPSILON))
    upper = float(max(np.quantile(train_target, 0.999), EPSILON * 10))
    dev_masks = {
        "full": np.ones((len(dev_target), 3), dtype="float32"),
        "random_25": random_mask(len(dev_target), 0.25, np.random.default_rng(seed + 25)),
        "random_50": random_mask(len(dev_target), 0.50, np.random.default_rng(seed + 50)),
    }

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for text, acoustic, price, quality, base_mask, log_target in loader:
            mask = base_mask.numpy()
            if dropout_probability > 0:
                mask = random_mask(len(mask), dropout_probability, np_rng)
            mask_tensor = torch.from_numpy(mask).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                text.to(device),
                acoustic.to(device),
                price.to(device),
                quality.to(device),
                mask_tensor,
            )
            loss = torch.mean((prediction - log_target.to(device)) ** 2)
            loss.backward()
            optimizer.step()

        dev_scores = []
        for mask in dev_masks.values():
            prediction = predict(
                model, dev_arrays, dev_quality, mask, device, lower, upper
            )
            dev_scores.append(qlike(dev_target, prediction))
        robust_score = float(np.mean(dev_scores))
        if robust_score < best_score - 1e-5:
            best_score = robust_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_dev_robust_qlike": best_score,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def scenario_masks(count: int, seed: int) -> dict[str, np.ndarray]:
    masks = {
        "full": np.ones((count, 3), dtype="float32"),
        "text_missing": np.tile([0, 1, 1], (count, 1)).astype("float32"),
        "acoustic_missing": np.tile([1, 0, 1], (count, 1)).astype("float32"),
        "price_missing": np.tile([1, 1, 0], (count, 1)).astype("float32"),
        "random_25": random_mask(count, 0.25, np.random.default_rng(seed + 25)),
        "random_50": random_mask(count, 0.50, np.random.default_rng(seed + 50)),
    }
    return masks


def aggregate(seed_results: list[dict]) -> dict:
    scenarios = seed_results[0]["scenarios"].keys()
    output = {}
    for scenario in scenarios:
        metric_names = seed_results[0]["scenarios"][scenario].keys()
        output[scenario] = {
            metric: {
                "mean": float(
                    np.mean([result["scenarios"][scenario][metric] for result in seed_results])
                ),
                "std": float(
                    np.std([result["scenarios"][scenario][metric] for result in seed_results])
                ),
            }
            for metric in metric_names
        }
    full_qlike = output["full"]["qlike"]["mean"]
    output["qlike_relative_degradation_percent"] = {
        scenario: float(
            100.0 * (values["qlike"]["mean"] - full_qlike) / max(full_qlike, EPSILON)
        )
        for scenario, values in output.items()
        if scenario != "qlike_relative_degradation_percent"
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA istendi fakat kullanılamıyor.")

    torch.set_num_threads(4)
    device = torch.device(args.device)
    root = Path(args.project_root).resolve()
    index = pd.read_csv(root / "data/processed/ahenk_dataset_index.csv")
    tabular = pd.read_parquet(root / "data/processed/ahenk_tabular_features.parquet")
    frame = tabular.merge(
        index[["call_id", "text_path"]], on="call_id", how="left", validate="one_to_one"
    ).sort_values(["call_date", "call_id"])
    acoustic_columns = [
        column
        for column in frame
        if column.startswith("ac_") or column == "acoustic_segment_count"
    ]
    price_columns = [column for column in frame if column.startswith("price_")]

    report_regimes = {}
    for split_column in ("temporal_split", "company_holdout_split"):
        print(f"Hazırlanıyor: {split_column}")
        train = frame[frame[split_column] == "train"].reset_index(drop=True)
        dev = frame[frame[split_column] == "dev"].reset_index(drop=True)
        test = frame[frame[split_column] == "test"].reset_index(drop=True)

        text_train, text_dev, text_test, text_info = fit_dense_text(
            root, train, dev, test, dimensions=128
        )
        acoustic_train, acoustic_dev, acoustic_test = fit_numeric(
            train, dev, test, acoustic_columns
        )
        price_train, price_dev, price_test = fit_numeric(
            train, dev, test, price_columns
        )
        train_quality, dev_quality, test_quality = quality_features(
            train, (train, dev, test)
        )
        train_arrays = (text_train, acoustic_train, price_train)
        dev_arrays = (text_dev, acoustic_dev, price_dev)
        test_arrays = (text_test, acoustic_test, price_test)
        dimensions = tuple(array.shape[1] for array in train_arrays)
        train_target = train[TARGET].to_numpy(float)
        dev_target = dev[TARGET].to_numpy(float)
        test_target = test[TARGET].to_numpy(float)
        lower = float(max(np.quantile(train_target, 0.001), EPSILON))
        upper = float(max(np.quantile(train_target, 0.999), EPSILON * 10))

        model_results = {}
        for model_name, specification in MODEL_SPECS.items():
            print(f"  Model: {model_name}")
            seed_results = []
            for seed in (3407, 2026, 2709):
                model, training_info = train_model(
                    specification["mode"],
                    specification["dropout"],
                    dimensions,
                    train_arrays,
                    dev_arrays,
                    train_quality,
                    dev_quality,
                    train_target,
                    dev_target,
                    device,
                    seed,
                    args.epochs,
                    args.patience,
                )
                scenarios = {}
                for scenario, mask in scenario_masks(len(test), seed).items():
                    prediction = predict(
                        model,
                        test_arrays,
                        test_quality,
                        mask,
                        device,
                        lower,
                        upper,
                    )
                    scenarios[scenario] = metric_set(test_target, prediction)
                seed_results.append(
                    {"seed": seed, "training": training_info, "scenarios": scenarios}
                )
            model_results[model_name] = {
                "mode": specification["mode"],
                "training_modality_dropout": specification["dropout"],
                "seed_results": seed_results,
                "aggregate": aggregate(seed_results),
            }

        report_regimes[split_column] = {
            "train_calls": len(train),
            "dev_calls": len(dev),
            "test_calls": len(test),
            "modality_dimensions": dict(zip(MODALITIES, dimensions, strict=True)),
            "text_preprocessing": text_info,
            "models": model_results,
        }

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "selection_metric": "mean dev QLIKE over full, random_25 and random_50",
        "synthetic_missingness_only": True,
        "test_used_for_model_selection": False,
        "future_prices_used_as_inputs": False,
        "human_or_expert_labels_used": False,
        "regimes": report_regimes,
        "status": "PASS",
    }
    output = root / "reports/models/ahenk_pilot.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Rapor: {output}")


if __name__ == "__main__":
    main()