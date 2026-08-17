"""Train/evaluate PhyNDR on the 720-assignment utilization dataset."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phyndr.config import FeatureConfig, ModelConfig  # noqa: E402
from phyndr.data.ndr720 import NDR720Dataset, collate_ndr720  # noqa: E402
from phyndr.losses.utilization import UtilizationLoss  # noqa: E402
from phyndr.models import PhyNDR  # noqa: E402


CASE = "zero-riscy_freq_200_mp_4_fpu_70_fpa_1.5_p_4_fi_ap"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "datasets" / "phyndr_ndr_720_v1")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "code" / "PhyNDR" / "baseline" / CASE / "phyndr_available_inputs_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_ndr720,
        generator=generator,
    )


def metrics(prediction: np.ndarray, target: np.ndarray, baseline: np.ndarray):
    error = prediction - target
    baseline_error = baseline - target
    response_pred = prediction - baseline
    response_true = target - baseline
    result = {}
    for channel, name in enumerate(("h", "v")):
        e = error[..., channel].reshape(-1)
        be = baseline_error[..., channel].reshape(-1)
        yp = prediction[..., channel].reshape(-1)
        yt = target[..., channel].reshape(-1)
        rp = response_pred[..., channel].reshape(-1)
        rt = response_true[..., channel].reshape(-1)
        denom = float(np.sum((yt - yt.mean()) ** 2))
        response_denom = float(np.sum((rt - rt.mean()) ** 2))
        result[f"mae_{name}"] = float(np.mean(np.abs(e)))
        result[f"rmse_{name}"] = float(np.sqrt(np.mean(e**2)))
        result[f"r2_{name}"] = float(1.0 - np.sum(e**2) / denom) if denom > 0 else float("nan")
        result[f"baseline_mae_{name}"] = float(np.mean(np.abs(be)))
        result[f"mae_improvement_vs_baseline_{name}"] = float(1.0 - np.mean(np.abs(e)) / max(np.mean(np.abs(be)), 1e-12))
        result[f"response_r2_{name}"] = float(1.0 - np.sum((rp - rt) ** 2) / response_denom) if response_denom > 0 else float("nan")
    result["mae"] = float(np.mean(np.abs(error)))
    result["rmse"] = float(np.sqrt(np.mean(error**2)))
    result["baseline_mae"] = float(np.mean(np.abs(baseline_error)))
    result["mae_improvement_vs_baseline"] = float(1.0 - result["mae"] / max(result["baseline_mae"], 1e-12))
    active = np.abs(response_true) > 1.0e-5
    result["active_response_mae"] = float(np.mean(np.abs(error[active]))) if active.any() else float("nan")
    return result


@torch.inference_mode()
def evaluate(model, data_loader, device):
    model.eval()
    predictions, targets, baselines, indices = [], [], [], []
    for graph, target, baseline, sample_index in data_loader:
        graph = graph.to(device)
        output = model(graph)["partition_utilization"].cpu().numpy()
        batch_size = len(sample_index)
        predictions.append(output.reshape(batch_size, 10, 2))
        targets.append(target.numpy().reshape(batch_size, 10, 2))
        baselines.append(baseline.numpy().reshape(batch_size, 10, 2))
        indices.append(sample_index.numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    baseline = np.concatenate(baselines)
    index = np.concatenate(indices)
    return metrics(prediction, target, baseline), prediction, target, baseline, index


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    datasets = {
        split: NDR720Dataset(args.dataset_root, args.baseline_root, split)
        for split in ("train", "validation", "test")
    }
    loaders = {
        "train": loader(datasets["train"], args.batch_size, True, args.seed),
        "validation": loader(datasets["validation"], args.batch_size, False, args.seed),
        "test": loader(datasets["test"], args.batch_size, False, args.seed),
    }
    config = ModelConfig(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        features=FeatureConfig(partition_state_dim=12),
    )
    model = PhyNDR(config).to(device)
    criterion = UtilizationLoss(args.loss, args.response_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1.0e-5)

    history = []
    best_value, best_epoch, stale = float("inf"), -1, 0
    checkpoint_path = args.output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = []
        for graph, target, baseline, _ in loaders["train"]:
            graph, target, baseline = graph.to(device), target.to(device), baseline.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(graph)
            losses = criterion(output, target, baseline)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            totals.append(float(losses["total"].detach().cpu()))
        validation, *_ = evaluate(model, loaders["validation"], device)
        selection = validation["active_response_mae"]
        scheduler.step(selection)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            "validation_mae": validation["mae"],
            "validation_active_response_mae": selection,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if selection < best_value - 1.0e-7:
            best_value, best_epoch, stale = selection, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": config.__dict__, "args": vars(args), "epoch": epoch}, checkpoint_path)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        if stale >= args.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    results = {}
    for split in ("train", "validation", "test"):
        split_metrics, prediction, target, baseline, index = evaluate(model, loaders[split], device)
        results[split] = split_metrics
        np.savez_compressed(
            args.output_dir / f"{split}_predictions.npz",
            sample_index=index,
            prediction=prediction,
            target=target,
            baseline=baseline,
        )
    summary = {
        "status": "complete",
        "best_epoch": best_epoch,
        "selection_metric": "validation active_response_mae",
        "config": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "loss": args.loss,
            "response_weight": args.response_weight,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "metrics": results,
        "interpretation": "A useful model must have positive MAE improvement versus the baseline-only predictor and positive response R2.",
    }
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
