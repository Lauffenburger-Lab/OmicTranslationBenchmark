from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from autotransop import (
    AdversaryConfig,
    AutoTransOP,
    AutoTransOPConfig,
    AutoTransOPTrainer,
    DomainConfig,
    DomainEffectConfig,
    DomainTensorData,
    HeadConfig,
    MLPConfig,
    MutualInformationConfig,
    PairRegularizerConfig,
    PriorConfig,
    TrainingConfig,
)
from autotransop.evaluation import (
    compare_per_feature_to_reference,
    per_feature_performance,
    plot_per_feature_performance_scatter,
    reconstruct_array,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a package-based AutoTransOP model on the HIV serology example."
    )
    parser.add_argument("--hiv-root", type=Path, default=Path("..") / ".." / "HIV_translation")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "package_examples" / "hiv_13_19")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--human-rows", type=int, default=323)
    parser.add_argument("--max-missing-fraction", type=float, default=0.05)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--protection-threshold", type=float, default=1.0)
    parser.add_argument("--flip-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distance-weight", type=float, default=4.0)
    parser.add_argument("--mi-weight", type=float, default=10.0)
    parser.add_argument("--species-head-weight", type=float, default=10.0)
    parser.add_argument("--species-adversary-weight", type=float, default=50.0)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def nan_cosine_distance(x, y, *, missing_values=np.nan) -> float:
    x_missing = np.isnan(x) if np.isnan(missing_values) else x == missing_values
    y_missing = np.isnan(y) if np.isnan(missing_values) else y == missing_values
    keep = ~(x_missing | y_missing)
    if keep.sum() == 0:
        return 1.0
    x_keep = x[keep]
    y_keep = y[keep]
    denom = np.linalg.norm(x_keep) * np.linalg.norm(y_keep)
    if denom <= 0:
        return 1.0
    similarity = float(np.dot(x_keep, y_keep) / denom)
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def select_and_impute(frame: pd.DataFrame, max_missing_fraction: float, n_neighbors: int) -> tuple[np.ndarray, list[str]]:
    missing_fraction = frame.isna().mean(axis=0)
    features = missing_fraction[missing_fraction <= max_missing_fraction].index.tolist()
    if not features:
        raise ValueError("No features remain after missing-value filtering.")

    values = frame.loc[:, features].to_numpy(dtype=np.float32)
    if np.isnan(values).any():
        neighbors = min(max(1, n_neighbors), max(1, values.shape[0] - 1))
        imputer = KNNImputer(n_neighbors=neighbors, weights="distance", metric=nan_cosine_distance)
        values = imputer.fit_transform(values).astype(np.float32)
    return values, features


def binary_protection(values: pd.Series, threshold: float, flip_labels: bool) -> np.ndarray:
    labels = (pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy() >= threshold).astype(np.int64)
    if flip_labels:
        labels = 1 - labels
    return labels


def split_indices(labels: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    counts = pd.Series(labels).value_counts()
    stratify = labels if len(counts) > 1 and int(counts.min()) >= 2 else None
    return train_test_split(np.arange(labels.shape[0]), test_size=test_size, random_state=seed, stratify=stratify)


def hidden_layers(input_dim: int) -> list[int]:
    if input_dim >= 128:
        return [128, 64]
    if input_dim >= 64:
        return [64]
    return [max(16, input_dim)]


def make_config(args: argparse.Namespace, human_dim: int, nhp_dim: int) -> AutoTransOPConfig:
    heads = []
    if args.species_head_weight > 0:
        heads.append(
            HeadConfig(
                "species_composed",
                target_key="species",
                latent_space="composed",
                num_outputs=2,
                weight=args.species_head_weight,
            )
        )

    adversaries = []
    if args.species_adversary_weight > 0:
        adversaries.append(
            AdversaryConfig(
                "species_global",
                target_key="species",
                latent_space="global",
                num_classes=2,
                weight=args.species_adversary_weight,
            )
        )

    return AutoTransOPConfig(
        version="v2",
        latent_dim=args.latent_dim,
        domains=[
            DomainConfig(
                "human",
                input_dim=human_dim,
                encoder=MLPConfig(hidden_layers(human_dim), dropout=0.3),
                decoder=MLPConfig(list(reversed(hidden_layers(human_dim))), dropout=0.2),
            ),
            DomainConfig(
                "nhp",
                input_dim=nhp_dim,
                encoder=MLPConfig(hidden_layers(nhp_dim), dropout=0.3),
                decoder=MLPConfig(list(reversed(hidden_layers(nhp_dim))), dropout=0.2),
            ),
        ],
        domain_effect=DomainEffectConfig(mode="vector", dropout=0.5, l2=1e-6),
        pair_regularizers=[
            PairRegularizerConfig(
                "protection_alignment",
                label_key="protection",
                metrics=("euclidean", "cosine"),
                latent_space="global",
                weight=args.distance_weight,
                metric_weights={"euclidean": 1.0, "cosine": 1.0},
                defined_key="defined",
                cross_domain_only=True,
            )
        ],
        mutual_information=MutualInformationConfig(
            enabled=args.mi_weight > 0,
            label_key="protection",
            latent_space="global",
            weight=args.mi_weight,
            defined_key="defined",
            cross_domain_only=True,
        ),
        heads=heads,
        adversaries=adversaries,
        prior=PriorConfig(enabled=args.prior_weight > 0, weight=args.prior_weight),
    )


def load_hiv_data(args: argparse.Namespace) -> dict:
    data_dir = args.hiv_root / "preprocessing" / "preprocessed_data"
    human_x_raw = pd.read_csv(data_dir / "XHPX2008.csv")
    human_y_raw = pd.read_csv(data_dir / "YHPX2008.csv")
    nhp_x_raw = pd.read_csv(data_dir / "XNHP_13_19.csv")
    nhp_y_raw = pd.read_csv(data_dir / "YNHP_13_19.csv")

    human_n = min(args.human_rows, len(human_x_raw), len(human_y_raw))
    nhp_n = min(len(nhp_x_raw), len(nhp_y_raw))
    human_x_raw = human_x_raw.iloc[:human_n, :].reset_index(drop=True)
    human_y_raw = human_y_raw.iloc[:human_n, :].reset_index(drop=True)
    nhp_x_raw = nhp_x_raw.iloc[:nhp_n, :].reset_index(drop=True)
    nhp_y_raw = nhp_y_raw.iloc[:nhp_n, :].reset_index(drop=True)

    human_x, human_features = select_and_impute(human_x_raw, args.max_missing_fraction, args.knn_neighbors)
    nhp_x, nhp_features = select_and_impute(nhp_x_raw, args.max_missing_fraction, args.knn_neighbors)

    human_label_col = "protection" if "protection" in human_y_raw.columns else human_y_raw.columns[0]
    nhp_label_col = "protection" if "protection" in nhp_y_raw.columns else nhp_y_raw.columns[0]
    human_y = binary_protection(human_y_raw[human_label_col], args.protection_threshold, args.flip_labels)
    nhp_y = binary_protection(nhp_y_raw[nhp_label_col], args.protection_threshold, args.flip_labels)

    human_defined = np.ones(human_n, dtype=np.int64)
    if "protection_defined" in nhp_y_raw.columns:
        nhp_defined = pd.to_numeric(nhp_y_raw["protection_defined"], errors="coerce").fillna(1).to_numpy()
        nhp_defined = (nhp_defined > 0).astype(np.int64)
    else:
        nhp_defined = np.ones(nhp_n, dtype=np.int64)

    human_train, human_val = split_indices(human_y, args.test_size, args.seed)
    nhp_train, nhp_val = split_indices(nhp_y, args.test_size, args.seed + 1)
    return {
        "human_x": human_x,
        "human_y": human_y,
        "human_defined": human_defined,
        "human_features": human_features,
        "human_train": human_train,
        "human_val": human_val,
        "nhp_x": nhp_x,
        "nhp_y": nhp_y,
        "nhp_defined": nhp_defined,
        "nhp_features": nhp_features,
        "nhp_train": nhp_train,
        "nhp_val": nhp_val,
    }


def domain_data(x: np.ndarray, y: np.ndarray, defined: np.ndarray, species: int, indices: np.ndarray) -> DomainTensorData:
    return DomainTensorData.from_arrays(
        x[indices],
        labels={
            "protection": y[indices].astype(np.int64),
            "defined": defined[indices].astype(np.int64),
            "species": np.full(indices.shape[0], species, dtype=np.int64),
        },
    )


def save_feature_evaluation(
    *,
    output_dir: Path,
    stem: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    features: list[str],
    shuffles: int,
    rng: np.random.Generator,
    make_plot: bool,
) -> float:
    frames = [per_feature_performance(y_true, y_pred, feature_names=features, set_name="validation")]
    for _ in range(max(0, shuffles)):
        frames.append(
            per_feature_performance(
                y_true,
                y_pred[rng.permutation(y_pred.shape[0])],
                feature_names=features,
                set_name="shuffled",
            )
        )
    performance = pd.concat(frames, ignore_index=True)
    comparison = compare_per_feature_to_reference(performance)
    performance.to_csv(output_dir / f"{stem}_per_feature_performance.csv", index=False)
    comparison.to_csv(output_dir / f"{stem}_per_feature_comparison.csv", index=False)
    if make_plot and not comparison.empty:
        plot_per_feature_performance_scatter(
            comparison,
            output_path=str(output_dir / f"{stem}_per_feature_scatter.png"),
            title=f"{stem} per-feature performance",
        )
    return float(frames[0]["r"].mean(skipna=True))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_hiv_data(args)

    config = make_config(args, human_dim=data["human_x"].shape[1], nhp_dim=data["nhp_x"].shape[1])
    model = AutoTransOP(config)
    trainer = AutoTransOPTrainer(
        model,
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        ),
    )
    history = trainer.fit(
        {
            "human": domain_data(data["human_x"], data["human_y"], data["human_defined"], 1, data["human_train"]),
            "nhp": domain_data(data["nhp_x"], data["nhp_y"], data["nhp_defined"], 0, data["nhp_train"]),
        }
    )

    history_frame = history.to_frame()
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    history_frame.to_csv(args.output_dir / "history.csv", index=False)

    eval_device = str(trainer.device)
    human_pred = reconstruct_array(
        model,
        data["human_x"][data["human_val"]],
        "human",
        batch_size=args.batch_size,
        device=eval_device,
    )
    nhp_pred = reconstruct_array(
        model,
        data["nhp_x"][data["nhp_val"]],
        "nhp",
        batch_size=args.batch_size,
        device=eval_device,
    )

    rng = np.random.default_rng(args.seed)
    human_mean_r = save_feature_evaluation(
        output_dir=args.output_dir,
        stem="human_reconstruction",
        y_true=data["human_x"][data["human_val"]],
        y_pred=human_pred,
        features=data["human_features"],
        shuffles=args.shuffles,
        rng=rng,
        make_plot=args.plot,
    )
    nhp_mean_r = save_feature_evaluation(
        output_dir=args.output_dir,
        stem="nhp_reconstruction",
        y_true=data["nhp_x"][data["nhp_val"]],
        y_pred=nhp_pred,
        features=data["nhp_features"],
        shuffles=args.shuffles,
        rng=rng,
        make_plot=args.plot,
    )

    print(f"Saved HIV AutoTransOP outputs to {args.output_dir.resolve()}")
    print(f"Mean validation Pearson r: human={human_mean_r:.3f}, nhp={nhp_mean_r:.3f}")


if __name__ == "__main__":
    main()
