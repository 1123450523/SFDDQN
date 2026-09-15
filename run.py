from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


BEGIN_DATE = "2010-01-01"
VAL_START = "2020-01-01"
SPLIT_POINT = "2022-01-01"
END_DATE = "2025-12-31"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SFDDQN algorithmic trading experiments."
    )

    parser.add_argument(
        "--experiment",
        choices=["standard", "mixed"],
        default="standard",
    )
    parser.add_argument(
        "--model",
        choices=["sfddqn", "ddqn"],
        default="sfddqn",
    )
    parser.add_argument(
        "--encoder",
        choices=["tsmixer", "mlp"],
        default="tsmixer",
    )
    parser.add_argument(
        "--mode",
        choices=["selection", "final"],
        default="final",
    )

    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--dataset", type=str)
    dataset_group.add_argument("--datasets", nargs="+", type=str)

    parser.add_argument("--sources", nargs="+", type=str)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--target", type=str)
    target_group.add_argument("--targets", nargs="+", type=str)

    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 17, 23, 31, 42],
    )

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--source-episodes-per-env", type=int, default=60)
    parser.add_argument("--target-episodes", type=int, default=20)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--replay-size", type=int, default=64)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--transaction-cost", type=float, default=0.0001)

    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.episodes <= 0:
        parser.error("--episodes must be positive.")
    if args.source_episodes_per_env <= 0:
        parser.error("--source-episodes-per-env must be positive.")
    if args.target_episodes <= 0:
        parser.error("--target-episodes must be positive.")
    if args.window <= 0:
        parser.error("--window must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.replay_size <= 0:
        parser.error("--replay-size must be positive.")
    if args.target_update <= 0:
        parser.error("--target-update must be positive.")
    if args.n_step <= 0:
        parser.error("--n-step must be positive.")
    if args.gamma < 0:
        parser.error("--gamma must be non-negative.")
    if args.transaction_cost < 0:
        parser.error("--transaction-cost must be non-negative.")

    if args.experiment == "standard":
        if not args.dataset and not args.datasets:
            parser.error("Standard experiments require --dataset or --datasets.")
        if args.sources or args.target or args.targets:
            parser.error(
                "--sources, --target and --targets are only valid for mixed experiments."
            )
    else:
        if args.mode != "final":
            parser.error("Mixed-source experiments currently support --mode final only.")
        if args.model != "sfddqn" or args.encoder != "tsmixer":
            parser.error(
                "Mixed-source experiments require "
                "--model sfddqn --encoder tsmixer."
            )
        if not args.sources:
            parser.error("Mixed-source experiments require --sources.")
        if not args.target and not args.targets:
            parser.error("Mixed-source experiments require --target or --targets.")
        if args.dataset or args.datasets:
            parser.error(
                "--dataset and --datasets are only valid for standard experiments."
            )


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seed is not None:
        return [args.seed]
    return list(args.seeds)


def resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.target is not None:
        return [args.target]
    return list(args.targets)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_split_config(mode: str) -> dict:
    return {
        "begin_date": BEGIN_DATE,
        "end_date": END_DATE,
        "split_point": SPLIT_POINT,
        "val_start": VAL_START if mode == "selection" else None,
    }


def load_trainer_class(model: str, encoder: str):
    if encoder == "tsmixer" and model == "sfddqn":
        from TSMixer.Train_Shared import Train
        return Train, "TSMixer-SFDDQN"

    if encoder == "tsmixer" and model == "ddqn":
        from TSMixer.Train_DirectQ import Train
        return Train, "TSMixer-DDQN"

    if encoder == "mlp" and model == "sfddqn":
        from MLP.Train_Shared import Train
        return Train, "MLP-SFDDQN"

    if encoder == "mlp" and model == "ddqn":
        from MLP.Train_Direct import Train
        return Train, "MLP-DDQN"

    raise ValueError(f"Unsupported model configuration: {model}/{encoder}")


def build_loader(dataset: str, mode: str):
    from RL.DataLoader import YahooFinanceDataLoader

    split = get_split_config(mode)

    return YahooFinanceDataLoader(
        dataset_name=dataset,
        split_point=split["split_point"],
        begin_date=split["begin_date"],
        end_date=split["end_date"],
        load_from_file=False,
        val_start=split["val_start"],
    )


def build_state(
    data,
    args: argparse.Namespace,
):
    from RL.State import State

    return State(
        data=data,
        action_name="action",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        gamma=args.gamma,
        n_step=args.n_step,
        batch_size=args.batch_size,
        window_size=args.window,
        transaction_cost=args.transaction_cost,
    )


def build_standard_environments(
    loader,
    args: argparse.Namespace,
):
    train_env = build_state(loader.data_train, args)
    val_env = (
        build_state(loader.data_val, args)
        if loader.data_val is not None
        else None
    )
    test_env = build_state(loader.data_test, args)

    return train_env, val_env, test_env


def build_trainer(
    trainer_class,
    loader,
    train_env,
    val_env,
    test_env,
    dataset: str,
    args: argparse.Namespace,
    experiment_dir: Path,
):
    return trainer_class(
        data_loader=loader,
        data_train=train_env,
        data_val=val_env,
        data_test=test_env,
        dataset_name=dataset,
        window_size=args.window,
        transaction_cost=args.transaction_cost,
        BATCH_SIZE=args.batch_size,
        GAMMA=args.gamma,
        ReplayMemorySize=args.replay_size,
        TARGET_UPDATE=args.target_update,
        n_step=args.n_step,
        experiment_dir=experiment_dir,
    )


def build_mixed_trainer(
    loader,
    train_env,
    test_env,
    dataset: str,
    args: argparse.Namespace,
    experiment_dir: Path,
):
    from TSMixer.Train_Mixed import TrainMixed

    return TrainMixed(
        data_loader=loader,
        data_train=train_env,
        data_test=test_env,
        dataset_name=dataset,
        window_size=args.window,
        transaction_cost=args.transaction_cost,
        BATCH_SIZE=args.batch_size,
        GAMMA=args.gamma,
        ReplayMemorySize=args.replay_size,
        TARGET_UPDATE=args.target_update,
        n_step=args.n_step,
        experiment_dir=experiment_dir,
    )


def save_evaluation(
    evaluation,
    loader,
    split_name: str,
    seed: int,
    run_dir: Path,
) -> dict:
    metrics = asdict(evaluation.compute_metrics())
    metrics_row = {"seed": seed, **metrics}

    pd.DataFrame([metrics_row]).to_csv(
        run_dir / "metrics.csv",
        index=False,
    )

    timeseries = evaluation.data.copy()

    if split_name == "val":
        dated_data = loader.data_val_with_date
    else:
        dated_data = loader.data_test_with_date

    if dated_data is not None and len(dated_data) == len(timeseries):
        timeseries.insert(
            0,
            "Date",
            pd.DatetimeIndex(dated_data.index).astype(str),
        )

    timeseries.to_csv(
        run_dir / f"{split_name}_timeseries.csv",
        index=False,
    )

    return metrics_row



def print_config(args: argparse.Namespace, seeds: list[int]) -> None:
    print()
    print("=" * 60)
    print("SFDDQN Experiment")
    print("=" * 60)
    print(f"Experiment       : {args.experiment}")
    print(f"Model            : {args.model.upper()}")
    print(f"Encoder          : {args.encoder.upper()}")
    print(f"Mode             : {args.mode}")
    print(f"Seeds            : {seeds}")
    print(f"Window           : {args.window}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Gamma            : {args.gamma}")
    print(f"Replay size      : {args.replay_size}")
    print(f"Target update    : {args.target_update}")
    print(f"Reward horizon   : {args.n_step}")
    print(f"Transaction cost : {args.transaction_cost}")

    if args.experiment == "standard":
        datasets = [args.dataset] if args.dataset else args.datasets
        print(f"Episodes         : {args.episodes}")
        print(f"Datasets         : {datasets}")
    else:
        targets = resolve_targets(args)
        print(f"Sources          : {args.sources}")
        print(f"Targets          : {targets}")
        print(f"Source episodes  : {args.source_episodes_per_env} per source")
        print(f"Target episodes  : {args.target_episodes} per target")

    print("=" * 60)
    print()


def run_standard(
    args: argparse.Namespace,
    seeds: list[int],
) -> None:
    trainer_class, model_name = load_trainer_class(
        args.model,
        args.encoder,
    )
    results_root = Path("Results")
    initial_capital = 500000
    datasets = [args.dataset] if args.dataset else args.datasets
    split_name = "val" if args.mode == "selection" else "test"

    for dataset in datasets:
        dataset_dir = results_root / args.mode / model_name / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            print(f"[{dataset}] seed={seed}")
            set_seed(seed)

            run_dir = dataset_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            loader = build_loader(dataset, args.mode)
            train_env, val_env, test_env = build_standard_environments(
                loader,
                args,
            )

            trainer = build_trainer(
                trainer_class=trainer_class,
                loader=loader,
                train_env=train_env,
                val_env=val_env,
                test_env=test_env,
                dataset=dataset,
                args=args,
                experiment_dir=run_dir,
            )

            trainer.train(num_episodes=args.episodes)

            evaluation = trainer.test(
                initial_investment=initial_capital,
                test_type=split_name,
            )

            metrics = save_evaluation(
                evaluation=evaluation,
                loader=loader,
                split_name=split_name,
                seed=seed,
                run_dir=run_dir,
            )
            print(
                f"CR={metrics['CR']:.4f}%  "
                f"AR={metrics['AR']:.4f}%  "
                f"Sharpe={metrics['Sharpe']:.4f}  "
                f"MDD={metrics['MDD']:.4f}%"
            )

        print()


def build_mixed_source_data(
    sources: list[str],
    args: argparse.Namespace,
):
    loaders = []
    envs = []

    for source in sources:
        loader = build_loader(source, "final")
        loaders.append(loader)
        envs.append(build_state(loader.data_train, args))

    return loaders, envs


def run_mixed(
    args: argparse.Namespace,
    seeds: list[int],
) -> None:
    targets = resolve_targets(args)
    results_root = Path("Results")
    initial_capital = 500000
    source_name = "-".join(args.sources)
    source_root = (
        results_root
        / "final"
        / "TSMixer-SFDDQN-Mixed"
        / f"{source_name}_source"
    )
    source_root.mkdir(parents=True, exist_ok=True)
    target_dirs = {}

    for target in targets:
        target_dir = (
            results_root
            / "final"
            / "TSMixer-SFDDQN-Mixed"
            / f"{source_name}_to_{target}"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target_dirs[target] = target_dir

    for seed in seeds:
        print(f"[mixed source: {', '.join(args.sources)}] seed={seed}")
        set_seed(seed)

        source_run_dir = source_root / f"seed_{seed}"
        source_run_dir.mkdir(parents=True, exist_ok=True)

        source_loaders, source_envs = build_mixed_source_data(
            args.sources,
            args,
        )
        source_loader = source_loaders[0]
        source_test_env = build_state(source_loader.data_test, args)

        source_trainer = build_mixed_trainer(
            loader=source_loader,
            train_env=source_envs[0],
            test_env=source_test_env,
            dataset=source_name,
            args=args,
            experiment_dir=source_run_dir,
        )
        source_trainer.set_train_envs(
            source_envs,
            env_names=args.sources,
        )
        source_trainer.train_per_env(
            episodes_per_env=args.source_episodes_per_env,
        )
        pretrained_state = source_trainer.get_pretrained_policy_state(
            to_cpu=True,
        )

        for target in targets:
            print(f"[mixed target: {target}] seed={seed}")
            set_seed(seed)

            target_run_dir = target_dirs[target] / f"seed_{seed}"
            target_run_dir.mkdir(parents=True, exist_ok=True)

            target_loader = build_loader(target, "final")
            target_train_env = build_state(target_loader.data_train, args)
            target_test_env = build_state(target_loader.data_test, args)

            target_trainer = build_mixed_trainer(
                loader=target_loader,
                train_env=target_train_env,
                test_env=target_test_env,
                dataset=target,
                args=args,
                experiment_dir=target_run_dir,
            )
            target_trainer.set_target_env(
                target_train_env,
                target_name=target,
            )
            target_trainer.load_pretrained_policy_state(
                pretrained_state,
            )
            target_trainer.train(
                num_episodes=args.target_episodes,
            )

            evaluation = target_trainer.test(
                initial_investment=initial_capital,
                test_type="test",
            )

            metrics = save_evaluation(
                evaluation=evaluation,
                loader=target_loader,
                split_name="test",
                seed=seed,
                run_dir=target_run_dir,
            )

            print(
                f"CR={metrics['CR']:.4f}%  "
                f"AR={metrics['AR']:.4f}%  "
                f"Sharpe={metrics['Sharpe']:.4f}  "
                f"MDD={metrics['MDD']:.4f}%"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    seeds = resolve_seeds(args)
    print_config(args, seeds)

    if args.experiment == "standard":
        run_standard(args, seeds)
    else:
        run_mixed(args, seeds)


if __name__ == "__main__":
    main()
