from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    translate_array,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a package-based AutoTransOP model on the A375/HT29 L1000 landmark split."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "package_examples" / "l1000_a375_ht29_landmarks")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--version", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=292)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--mi-weight", type=float, default=1.0)
    parser.add_argument("--metrics", type=str, default="euclidean,cosine")
    parser.add_argument("--cell-head-weight", type=float, default=0.0)
    parser.add_argument("--cell-adversary-weight", type=float, default=0.0)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def split_path(split_dir: Path, name: str, fold: int) -> Path:
    path = split_dir / f"{name}_{fold}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_split(split_dir: Path, name: str, fold: int) -> pd.DataFrame:
    return pd.read_csv(split_path(split_dir, name, fold))


def matrix_for_signatures(cmap: pd.DataFrame, signature_ids: list[str]) -> np.ndarray:
    missing = [sig_id for sig_id in signature_ids if sig_id not in cmap.index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} signature ids were not found in the landmark matrix. First missing: {preview}")
    return cmap.loc[signature_ids].to_numpy(dtype=np.float32)


def domain_rows(
    cmap: pd.DataFrame,
    paired: pd.DataFrame,
    unpaired: pd.DataFrame,
    paired_sig_col: str,
) -> tuple[np.ndarray, list[str]]:
    signature_ids = paired[paired_sig_col].astype(str).tolist() + unpaired["sig_id"].astype(str).tolist()
    conditions = paired["conditionId"].astype(str).tolist() + unpaired["conditionId"].astype(str).tolist()
    return matrix_for_signatures(cmap, signature_ids), conditions


def paired_domain_rows(cmap: pd.DataFrame, paired: pd.DataFrame, paired_sig_col: str) -> np.ndarray:
    signature_ids = paired[paired_sig_col].astype(str).tolist()
    return matrix_for_signatures(cmap, signature_ids)


def encode_conditions(*condition_lists: list[str]) -> tuple[list[np.ndarray], dict[str, int]]:
    all_conditions = pd.Index([condition for values in condition_lists for condition in values])
    unique_conditions = sorted(all_conditions.unique())
    mapping = {condition: i for i, condition in enumerate(unique_conditions)}
    encoded = [np.asarray([mapping[condition] for condition in values], dtype=np.int64) for values in condition_lists]
    return encoded, mapping


def metrics_from_arg(value: str) -> tuple[str, ...]:
    metrics = tuple(item.strip() for item in value.split(",") if item.strip())
    if not metrics:
        raise ValueError("--metrics must include at least one metric.")
    return metrics


def make_config(args: argparse.Namespace, input_dim: int, metrics: tuple[str, ...]) -> AutoTransOPConfig:
    heads = []
    if args.cell_head_weight > 0:
        heads.append(
            HeadConfig(
                "cell_line_head",
                target_key="cell_line",
                latent_space="composed" if args.version == "v2" else "global",
                num_outputs=2,
                weight=args.cell_head_weight,
            )
        )

    adversaries = []
    if args.cell_adversary_weight > 0:
        adversaries.append(
            AdversaryConfig(
                "cell_line_global",
                target_key="cell_line",
                latent_space="global",
                num_classes=2,
                weight=args.cell_adversary_weight,
            )
        )

    return AutoTransOPConfig(
        version=args.version,
        latent_dim=args.latent_dim,
        domains=[
            DomainConfig(
                "a375",
                input_dim=input_dim,
                encoder=MLPConfig([640, 384], dropout=0.1, input_dropout=0.5),
                decoder=MLPConfig([384, 640], dropout=0.2),
                encoder_l2=0.01,
                decoder_l2=0.01,
            ),
            DomainConfig(
                "ht29",
                input_dim=input_dim,
                encoder=MLPConfig([640, 384], dropout=0.1, input_dropout=0.5),
                decoder=MLPConfig([384, 640], dropout=0.2),
                encoder_l2=0.01,
                decoder_l2=0.01,
            ),
        ],
        domain_effect=DomainEffectConfig(mode="vector", dropout=0.5, l2=1e-6),
        pair_regularizers=[
            PairRegularizerConfig(
                "condition_alignment",
                label_key="condition",
                metrics=metrics,
                latent_space="global",
                weight=args.distance_weight,
                cross_domain_only=True,
            )
        ]
        if args.distance_weight > 0
        else [],
        mutual_information=MutualInformationConfig(
            enabled=args.mi_weight > 0,
            label_key="condition",
            latent_space="global",
            weight=args.mi_weight,
            cross_domain_only=True,
        ),
        heads=heads,
        adversaries=adversaries,
        prior=PriorConfig(enabled=args.prior_weight > 0, weight=args.prior_weight),
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
    output_dir = args.output_dir / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.repo_root / "preprocessing" / "preprocessed_data"
    split_dir = data_dir / "10fold_validation_spit"
    cmap = pd.read_csv(data_dir / "cmap_landmarks_HT29_A375.csv", index_col=0)
    features = cmap.columns.astype(str).tolist()

    train_paired = read_split(split_dir, "train_paired", args.fold)
    train_a375 = read_split(split_dir, "train_a375", args.fold)
    train_ht29 = read_split(split_dir, "train_ht29", args.fold)
    val_paired = read_split(split_dir, "val_paired", args.fold)
    val_a375 = read_split(split_dir, "val_a375", args.fold)
    val_ht29 = read_split(split_dir, "val_ht29", args.fold)

    x_a375_train, conditions_a375 = domain_rows(cmap, train_paired, train_a375, "sig_id.x")
    x_ht29_train, conditions_ht29 = domain_rows(cmap, train_paired, train_ht29, "sig_id.y")
    (condition_a375, condition_ht29), condition_mapping = encode_conditions(conditions_a375, conditions_ht29)

    datasets = {
        "a375": DomainTensorData.from_arrays(
            x_a375_train,
            labels={
                "condition": condition_a375,
                "cell_line": np.zeros(x_a375_train.shape[0], dtype=np.int64),
            },
        ),
        "ht29": DomainTensorData.from_arrays(
            x_ht29_train,
            labels={
                "condition": condition_ht29,
                "cell_line": np.ones(x_ht29_train.shape[0], dtype=np.int64),
            },
        ),
    }

    metrics = metrics_from_arg(args.metrics)
    config = make_config(args, input_dim=cmap.shape[1], metrics=metrics)
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
    history = trainer.fit(datasets)

    history_frame = history.to_frame()
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    history_frame.to_csv(output_dir / "history.csv", index=False)
    pd.Series(condition_mapping, name="condition_code").to_csv(output_dir / "condition_mapping.csv")

    x_a375_val, _ = domain_rows(cmap, val_paired, val_a375, "sig_id.x")
    x_ht29_val, _ = domain_rows(cmap, val_paired, val_ht29, "sig_id.y")
    x_a375_paired_val = paired_domain_rows(cmap, val_paired, "sig_id.x")
    x_ht29_paired_val = paired_domain_rows(cmap, val_paired, "sig_id.y")

    eval_device = str(trainer.device)
    a375_reconstruction = reconstruct_array(model, x_a375_val, "a375", batch_size=args.batch_size, device=eval_device)
    ht29_reconstruction = reconstruct_array(model, x_ht29_val, "ht29", batch_size=args.batch_size, device=eval_device)
    a375_to_ht29 = translate_array(
        model,
        x_a375_paired_val,
        "a375",
        "ht29",
        batch_size=args.batch_size,
        device=eval_device,
    )
    ht29_to_a375 = translate_array(
        model,
        x_ht29_paired_val,
        "ht29",
        "a375",
        batch_size=args.batch_size,
        device=eval_device,
    )

    rng = np.random.default_rng(args.seed)
    summaries = {
        "a375_reconstruction": save_feature_evaluation(
            output_dir=output_dir,
            stem="a375_reconstruction",
            y_true=x_a375_val,
            y_pred=a375_reconstruction,
            features=features,
            shuffles=args.shuffles,
            rng=rng,
            make_plot=args.plot,
        ),
        "ht29_reconstruction": save_feature_evaluation(
            output_dir=output_dir,
            stem="ht29_reconstruction",
            y_true=x_ht29_val,
            y_pred=ht29_reconstruction,
            features=features,
            shuffles=args.shuffles,
            rng=rng,
            make_plot=args.plot,
        ),
        "a375_to_ht29_translation": save_feature_evaluation(
            output_dir=output_dir,
            stem="a375_to_ht29_translation",
            y_true=x_ht29_paired_val,
            y_pred=a375_to_ht29,
            features=features,
            shuffles=args.shuffles,
            rng=rng,
            make_plot=args.plot,
        ),
        "ht29_to_a375_translation": save_feature_evaluation(
            output_dir=output_dir,
            stem="ht29_to_a375_translation",
            y_true=x_a375_paired_val,
            y_pred=ht29_to_a375,
            features=features,
            shuffles=args.shuffles,
            rng=rng,
            make_plot=args.plot,
        ),
    }
    pd.Series(summaries, name="mean_validation_pearson_r").to_csv(output_dir / "summary.csv")

    print(f"Saved L1000 AutoTransOP outputs to {output_dir.resolve()}")
    for name, value in summaries.items():
        print(f"{name}: mean validation Pearson r={value:.3f}")


if __name__ == "__main__":
    main()
