"""Generate the deterministic 720-sample PhyNDR proxy dataset.

The baseline (partition, layer) congestion state is inferred by the trained
RouteGNN_v4 model.  Its UNet branch carries placement/macro/RUDY context and
the frozen Blockage Encoder is evaluated with exactly one routing layer at a
time.  The NDR response and the currently unavailable timing/power targets are
explicit synthetic proxies; they are never presented as OpenROAD labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


CASE = "zero-riscy_freq_200_mp_4_fpu_70_fpa_1.5_p_4_fi_ap"
ACTION_NAMES = np.asarray(["1W1S", "1W2S", "1W3S", "2W3S"])
WIDTH_RATIO = np.asarray([1.0, 1.0, 1.0, 2.0], dtype=np.float32)
SPACING_RATIO = np.asarray([1.0, 2.0, 3.0, 3.0], dtype=np.float32)
# Direct synthetic utilization response coefficients.  These are not capacity
# loss values and do not modify track capacity.
UTILIZATION_GAIN = np.asarray([0.0, 0.055, 0.125, 0.225], dtype=np.float32)
SEED = 20260814


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    repo = here.parents[4]
    project = here.parents[1]
    routegnn = repo / "code" / "RouteGNN_v4"
    baseline = repo / "code" / "PhyNDR" / "baseline" / CASE / "phyndr_available_inputs_v1"
    experiment = routegnn / "experiments" / "n14" / "routegnn_v4_unet_blockage_gated_200epoch"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=baseline)
    parser.add_argument(
        "--routegnn-dataset-root",
        type=Path,
        default=routegnn / "datasets" / "n14" / "routegnn_benchmark_v1_bin32x32",
    )
    parser.add_argument("--routegnn-root", type=Path, default=routegnn)
    parser.add_argument(
        "--routegnn-checkpoint",
        type=Path,
        default=experiment / "n14_routegnn_v4_unet_blockage_gated_200epoch_best.pkl",
    )
    parser.add_argument(
        "--blockage-checkpoint",
        type=Path,
        default=repo / "code" / "encode_blockage" / "experiments" / "n14_split_heads_rebuild_100epoch" / "best.pt",
    )
    parser.add_argument(
        "--direction-config",
        type=Path,
        default=repo / "code" / "encode_blockage" / "configs" / "n14_layer_directions.json",
    )
    parser.add_argument(
        "--encoder-file",
        type=Path,
        default=repo / "code" / "encode_blockage" / "model" / "encoder.py",
    )
    parser.add_argument("--output-dir", type=Path, default=project / "datasets" / "phyndr_ndr_720_v1")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reuse-baseline", action="store_true")
    return parser.parse_args()


def import_routegnn_helpers(routegnn_root: Path):
    sys.path.insert(0, str(routegnn_root.resolve()))
    from scripts.evaluate_single_metal_layer_case import (  # noqa: PLC0415
        attach_single_layer_features,
        build_routegnn,
    )
    from scripts.extract_blockage_gcell_features import (  # noqa: PLC0415
        load_frozen_encoder,
        make_patch,
        read_directions,
    )
    from scripts.train_test_single_case import forward_model, load_processed_case  # noqa: PLC0415

    return attach_single_layer_features, build_routegnn, load_frozen_encoder, make_patch, read_directions, forward_model, load_processed_case


def extract_layer_embedding(
    physical_path: Path,
    interface_path: Path,
    layer_name: str,
    direction_map: dict[str, int],
    encoder,
    make_patch,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], np.ndarray, int]:
    with np.load(interface_path, allow_pickle=True) as interface:
        grid_shape = tuple(int(v) for v in interface["h_net_density_grid"].shape)
        graph_bin_size = np.asarray(interface["graph_bin_size"], dtype=np.float64).reshape(2)
    with np.load(physical_path, allow_pickle=False) as archive:
        layer_names = tuple(str(v) for v in archive["layer_names"].tolist())
        layer_index = layer_names.index(layer_name)
        # Keep the encoder preprocessing identical to its training contract.
        # macro_region is already carried by RouteGNN's frozen UNet branch.
        physical = np.stack(
            [
                np.asarray(archive["routing_blockage"][layer_index : layer_index + 1], dtype=np.float32),
                np.asarray(archive["pg_vdd"][layer_index : layer_index + 1], dtype=np.float32),
                np.asarray(archive["pg_vss"][layer_index : layer_index + 1], dtype=np.float32),
            ],
            axis=1,
        )
    direction = int(direction_map[layer_name])
    indices = np.asarray([(x, y) for x in range(grid_shape[0]) for y in range(grid_shape[1])])
    patches = np.stack([make_patch(physical, int(x), int(y)) for x, y in indices])
    patch_tensor = torch.from_numpy(patches).to(device=device, dtype=torch.float32)
    directions = torch.full((len(indices), 1), direction, dtype=torch.long, device=device)
    mask = torch.ones_like(directions, dtype=torch.bool)
    with torch.inference_mode():
        _, details = encoder(patch_tensor, directions, mask, return_attention=True)
    h_map = details["horizontal_embedding"].cpu().numpy().astype(np.float32).reshape(*grid_shape, 64)
    v_map = details["vertical_embedding"].cpu().numpy().astype(np.float32).reshape(*grid_shape, 64)
    availability = np.zeros((*grid_shape, 2), dtype=bool)
    availability[..., 0] = direction in (0, 2)
    availability[..., 1] = direction in (1, 2)
    return h_map, v_map, availability, grid_shape, graph_bin_size, direction


def map_model_cells_to_partitions(interface_path: Path, cells_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(interface_path, allow_pickle=True) as interface:
        node_names = np.asarray(interface["node_names"], dtype=str)
    cells = pd.read_csv(cells_path)
    rows = cells.set_index("instance_name").reindex(node_names)
    if rows["partition_index"].isna().any():
        missing = node_names[rows["partition_index"].isna().to_numpy()][:5]
        raise ValueError(f"RouteGNN cells absent from baseline cells.csv: {missing.tolist()}")
    return (
        rows["partition_index"].to_numpy(dtype=np.int64),
        rows[["x", "y"]].to_numpy(dtype=np.float64),
        node_names,
    )


def aggregate_partition(values: np.ndarray, partition_index: np.ndarray, xy: np.ndarray, partitions: pd.DataFrame):
    means = np.empty(len(partitions), dtype=np.float32)
    p90 = np.empty(len(partitions), dtype=np.float32)
    direct = np.ones(len(partitions), dtype=bool)
    counts = np.zeros(len(partitions), dtype=np.int64)
    for p, row in partitions.iterrows():
        mask = partition_index == p
        counts[p] = int(mask.sum())
        if counts[p]:
            selected = values[mask]
        else:
            # R10 has no cell nodes.  Use the nearest 64 model-predicted cells,
            # and record that this is a fallback rather than a direct aggregate.
            center = np.asarray([row["center_x"], row["center_y"]], dtype=np.float64)
            nearest = np.argsort(np.sum((xy - center[None, :]) ** 2, axis=1))[:64]
            selected = values[nearest]
            direct[p] = False
        means[p] = float(np.mean(selected))
        p90[p] = float(np.quantile(selected, 0.90))
    return means, p90, direct, counts


def infer_baseline_layer_state(args: argparse.Namespace) -> dict[str, np.ndarray]:
    helpers = import_routegnn_helpers(args.routegnn_root)
    attach, build_routegnn, load_encoder, make_patch, read_directions, forward_model, load_case = helpers
    device = torch.device(args.device)
    interface_path = args.routegnn_dataset_root / "netlists" / CASE / "routegnn_interface.npz"
    physical_path = args.baseline_root / "physical_layers.npz"
    processed_path = args.routegnn_dataset_root / "processed" / CASE
    partitions = pd.read_csv(args.baseline_root / "partitions.csv").sort_values("partition_index").reset_index(drop=True)
    supply = pd.read_csv(args.baseline_root / "partition_layer_supply.csv").sort_values("r_index")
    layer_names = supply.drop_duplicates("layer_index").sort_values("layer_index")["layer_name"].tolist()
    direction_map = read_directions(args.direction_config)
    encoder, _ = load_encoder(args.encoder_file, args.blockage_checkpoint, device)
    model, model_args = build_routegnn(device, args.routegnn_checkpoint)
    _, _, _, graphs, _, _ = load_case(processed_path)
    cell_partition, cell_xy, node_names = map_model_cells_to_partitions(interface_path, args.baseline_root / "cells.csv")

    mean_pl = np.empty((len(partitions), len(layer_names)), dtype=np.float32)
    p90_pl = np.empty_like(mean_pl)
    direct_pl = np.empty_like(mean_pl, dtype=bool)
    cell_counts = np.empty_like(mean_pl, dtype=np.int64)
    direction_l = np.empty(len(layer_names), dtype=np.int64)
    for layer_index, layer_name in enumerate(layer_names):
        h_map, v_map, availability, grid_shape, graph_bin_size, direction = extract_layer_embedding(
            physical_path, interface_path, layer_name, direction_map, encoder, make_patch, device
        )
        direction_l[layer_index] = direction
        outputs = []
        with torch.inference_mode():
            for graph in graphs:
                attach(graph, h_map, v_map, availability, grid_shape, graph_bin_size)
                device_graph = graph.to(device)
                pred_log = forward_model(model, device_graph, model_args.add_pos)
                outputs.append(torch.expm1(pred_log).clamp_min(0).cpu().numpy())
        prediction = np.concatenate(outputs, axis=0)
        if prediction.shape[0] != node_names.shape[0]:
            raise ValueError("processed graph order is not aligned with routegnn_interface.npz")
        directed = prediction[:, 0 if direction == 0 else 1]
        means, p90, direct, counts = aggregate_partition(directed, cell_partition, cell_xy, partitions)
        mean_pl[:, layer_index] = means
        p90_pl[:, layer_index] = np.maximum(p90, means)
        direct_pl[:, layer_index] = direct
        cell_counts[:, layer_index] = counts
        print(f"predicted {layer_name}: mean=[{means.min():.6f}, {means.max():.6f}]", flush=True)
    return {
        "mean_pl": mean_pl,
        "p90_pl": p90_pl,
        "direct_pl": direct_pl,
        "cell_counts_pl": cell_counts,
        "direction_l": direction_l,
        "layer_names": np.asarray(layer_names),
    }


def create_assignments(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    families: list[str] = []
    # Exhaustive one-site coverage: every R(P,L) receives each non-baseline action.
    for r_index in range(60):
        for action in (1, 2, 3):
            row = np.zeros(60, dtype=np.int8)
            row[r_index] = action
            rows.append(row)
            families.append("one")

    seen: set[bytes] = {row.tobytes() for row in rows}

    def append_random(count: int, k: int, family: str) -> None:
        while sum(name == family for name in families) < count:
            row = np.zeros(60, dtype=np.int8)
            sites = rng.choice(60, size=k, replace=False)
            row[sites] = rng.integers(1, 4, size=k, dtype=np.int8)
            key = row.tobytes()
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            families.append(family)

    append_random(90, 2, "two")
    append_random(90, 3, "three")
    count_by_k = {4: 52, 5: 52, 6: 52, 7: 51, 8: 51, 9: 51, 10: 51}
    for k, count in count_by_k.items():
        append_random(count, k, f"multi_{k}")
    actions = np.stack(rows)
    family = np.asarray(families)
    if actions.shape != (720, 60):
        raise AssertionError(actions.shape)
    # Shuffle storage order while retaining deterministic identities.
    order = rng.permutation(len(actions))
    return actions[order], family[order]


def create_splits(family: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    split = np.full(len(family), "", dtype="<U10")
    strata = {
        "one": np.flatnonzero(family == "one"),
        "two": np.flatnonzero(family == "two"),
        "three": np.flatnonzero(family == "three"),
        "multi": np.flatnonzero(np.char.startswith(family, "multi_")),
    }
    for indices in strata.values():
        indices = rng.permutation(indices)
        n_train = int(round(0.8 * len(indices)))
        n_val = int(round(0.1 * len(indices)))
        split[indices[:n_train]] = "train"
        split[indices[n_train : n_train + n_val]] = "validation"
        split[indices[n_train + n_val :]] = "test"
    return split


def partition_adjacency(baseline_root: Path) -> list[tuple[int, int]]:
    table = pd.read_csv(baseline_root / "partition_adjacency.csv")
    pairs = set()
    for row in table.itertuples(index=False):
        a = int(getattr(row, "src_partition_index"))
        b = int(getattr(row, "dst_partition_index"))
        if a != b:
            pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def synthesize_utilization(
    actions: np.ndarray,
    mean_pl: np.ndarray,
    p90_pl: np.ndarray,
    supply: pd.DataFrame,
    adjacency: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_samples = len(actions)
    action_pl = actions.reshape(num_samples, 10, 6)
    mean_after = np.broadcast_to(mean_pl, (num_samples, *mean_pl.shape)).copy()
    p90_after = np.broadcast_to(p90_pl, (num_samples, *p90_pl.shape)).copy()

    demand = pd.read_csv(supply.attrs["baseline_root"] / "partition_net_demand.csv").sort_values("partition_index")
    demand_h = np.nan_to_num(demand["demand_H"].to_numpy(dtype=np.float64), nan=0.0)
    demand_v = np.nan_to_num(demand["demand_V"].to_numpy(dtype=np.float64), nan=0.0)
    demand_h /= max(np.quantile(demand_h[demand_h > 0], 0.9), 1.0)
    demand_v /= max(np.quantile(demand_v[demand_v > 0], 0.9), 1.0)
    directions = supply.drop_duplicates("layer_index").sort_values("layer_index")["preferred_direction"].to_numpy()
    sensitivity = np.empty((10, 6), dtype=np.float32)
    for p in range(10):
        for layer in range(6):
            d = demand_h[p] if directions[layer] == "H" else demand_v[p]
            sensitivity[p, layer] = np.clip(0.72 + 0.36 * d, 0.72, 1.25)

    local_delta = np.zeros_like(mean_after)
    for sample in range(num_samples):
        counts_per_partition = np.count_nonzero(action_pl[sample], axis=1)
        for p, layer in np.argwhere(action_pl[sample] > 0):
            action = int(action_pl[sample, p, layer])
            synergy = 1.0 + 0.06 * max(int(counts_per_partition[p]) - 1, 0)
            delta = UTILIZATION_GAIN[action] * (0.30 + 0.70 * mean_pl[p, layer])
            local_delta[sample, p, layer] += float(delta * sensitivity[p, layer] * synergy)
        # Same-layer boundary spill and same-partition adjacent-layer spill.
        for a, b in adjacency:
            local_delta[sample, a] += 0.10 * local_delta[sample, b]
            local_delta[sample, b] += 0.10 * local_delta[sample, a]
        original = local_delta[sample].copy()
        local_delta[sample, :, 1:] += 0.045 * original[:, :-1]
        local_delta[sample, :, :-1] += 0.045 * original[:, 1:]

    mean_after += local_delta
    p90_after += 1.35 * local_delta
    mean_after = np.clip(mean_after, 0.0, 2.5).astype(np.float32)
    p90_after = np.maximum(mean_after, np.clip(p90_after, 0.0, 3.0)).astype(np.float32)

    track = supply.sort_values("r_index")["track_number"].to_numpy(dtype=np.float64).reshape(10, 6)
    y_partition = np.empty((num_samples, 10, 2), dtype=np.float32)
    for channel, wanted in enumerate(("H", "V")):
        mask = directions == wanted
        weights = track[:, mask]
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
        y_partition[:, :, channel] = np.sum(mean_after[:, :, mask] * weights[None, :, :], axis=2)
    baseline_partition = np.empty((10, 2), dtype=np.float32)
    for channel, wanted in enumerate(("H", "V")):
        mask = directions == wanted
        weights = track[:, mask]
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
        baseline_partition[:, channel] = np.sum(mean_pl[:, mask] * weights, axis=1)
    return mean_after, p90_after, y_partition, baseline_partition


def create_static_inputs(args, baseline_state, baseline_partition, seed):
    baseline = args.baseline_root
    partitions = pd.read_csv(baseline / "partitions.csv").sort_values("partition_index").reset_index(drop=True)
    supply = pd.read_csv(baseline / "partition_layer_supply.csv").sort_values("r_index").reset_index(drop=True)
    demand = pd.read_csv(baseline / "partition_net_demand.csv").sort_values("partition_index").reset_index(drop=True)
    cells = pd.read_csv(baseline / "cells.csv")
    macros = pd.read_csv(baseline / "macros.csv")
    x_supply = supply[
        ["track_number", "total_blockage_rate", "default_width", "default_spacing", "pitch", "macro_obs_blockage_rate", "pg_vdd_blockage_rate", "pg_vss_blockage_rate"]
    ].to_numpy(dtype=np.float32)
    x_state_r = np.column_stack([baseline_state["mean_pl"].reshape(-1), baseline_state["p90_pl"].reshape(-1)]).astype(np.float32)
    direction = (supply["preferred_direction"].to_numpy() == "V").astype(np.int8)

    cell_count = cells.groupby("partition_index").size().reindex(range(10), fill_value=0).to_numpy()
    cell_area = cells.groupby("partition_index")["area"].sum().reindex(range(10), fill_value=0).to_numpy()
    macro_count = macros.groupby("partition_index").size().reindex(range(10), fill_value=0).to_numpy()
    macro_area = macros.groupby("partition_index")["area"].sum().reindex(range(10), fill_value=0).to_numpy()
    area = partitions["area"].to_numpy(dtype=np.float64)
    x_state_u = np.column_stack(
        [
            cell_count,
            cell_area / np.maximum(area, 1e-6),
            macro_count,
            macro_area / np.maximum(area, 1e-6),
            np.nan_to_num(demand["internal_net_count"].to_numpy()),
            np.nan_to_num(demand["cross_partition_net_count"].to_numpy()),
            baseline_partition[:, 0],
            baseline_partition[:, 1],
        ]
    ).astype(np.float32)
    x_demand_u = np.nan_to_num(
        demand[["demand_H", "demand_V", "hpwl_sum", "fanout_mean", "fanout_p90", "high_fanout_net_count"]].to_numpy(dtype=np.float32),
        nan=0.0,
    )

    rng = np.random.default_rng(seed + 2)
    density = cell_area / np.maximum(area, 1e-6)
    dynamic = (0.12 + 1.8 * density + 0.00002 * cell_count) * rng.uniform(0.94, 1.06, 10)
    leakage = (0.035 + 0.000004 * cell_area) * rng.uniform(0.96, 1.04, 10)
    power = np.column_stack([dynamic, leakage, dynamic + leakage, 1.15 * (dynamic + leakage)]).astype(np.float32)

    return {
        "x_supply": x_supply,
        "x_state_r": x_state_r,
        "direction_r": direction,
        "partition_index_r": supply["partition_index"].to_numpy(dtype=np.int16),
        "layer_index_r": supply["layer_index"].to_numpy(dtype=np.int8),
        "x_state_u": x_state_u,
        "x_demand_u": x_demand_u,
        "demand_h": x_demand_u[:, 0],
        "demand_v": x_demand_u[:, 1],
        "power_proxy_u": power,
        "partition_valid": demand["valid"].to_numpy(dtype=bool),
        "layer_aggregation_direct": baseline_state["direct_pl"].reshape(-1),
        "layer_aggregation_cell_count": baseline_state["cell_counts_pl"].reshape(-1),
    }


def create_critical_nets(args, seed):
    with np.load(args.baseline_root / "nets.npz", allow_pickle=True) as nets:
        names = np.asarray(nets["net_names"], dtype=str)
        hpwl = np.asarray(nets["hpwl"], dtype=np.float64)
        fanout = np.asarray(nets["fanout"], dtype=np.float64)
        span_h = np.asarray(nets["span_H"], dtype=np.float64)
        span_v = np.asarray(nets["span_V"], dtype=np.float64)
        incident = np.asarray(nets["incident_partition_count"], dtype=np.float64)
    score = np.log1p(np.maximum(hpwl, 0)) * np.log1p(np.maximum(fanout, 0)) * (1 + 0.15 * incident)
    chosen = np.argsort(score)[-256:][::-1]
    normalized = score[chosen] / max(score[chosen].max(), 1e-9)
    rng = np.random.default_rng(seed + 3)
    criticality = np.clip(0.55 + 0.43 * normalized + rng.normal(0, 0.015, len(chosen)), 0, 1)
    slack = -(0.015 + 0.22 * criticality + rng.normal(0, 0.008, len(chosen)))
    path_member = criticality >= np.quantile(criticality, 0.70)
    x = np.column_stack([slack, criticality, path_member, fanout[chosen], span_h[chosen], span_v[chosen], hpwl[chosen], incident[chosen]]).astype(np.float32)

    with np.load(args.baseline_root / "partition_net_incidence.npz", allow_pickle=False) as incidence:
        all_net = incidence["net_index"]
        mask = np.isin(all_net, chosen) & incidence["valid"]
        old_to_new = {int(net): i for i, net in enumerate(chosen)}
        critical_index = np.asarray([old_to_new[int(net)] for net in all_net[mask]], dtype=np.int32)
        part = incidence["partition_index"][mask].astype(np.int16)
        edge_x = np.column_stack(
            [incidence["pin_fraction_in_partition"][mask], incidence["local_span_H"][mask], incidence["local_span_V"][mask]]
        ).astype(np.float32)
    return {
        "critical_net_index": chosen.astype(np.int32),
        "critical_net_name": names[chosen],
        "x_critical": x,
        "incidence_partition_index": part,
        "incidence_critical_index": critical_index,
        "incidence_x": edge_x,
    }


def create_aux_targets(actions, y_partition, baseline_partition, seed):
    rng = np.random.default_rng(seed + 4)
    delta = y_partition - baseline_partition[None, :, :]
    changed = np.count_nonzero(actions, axis=1)
    stress = delta.sum(axis=(1, 2))
    peak = delta.max(axis=(1, 2))
    baseline_chip = np.asarray([24.0, -0.086, -5.40], dtype=np.float32)
    drc = np.rint(baseline_chip[0] + 16.0 * stress + rng.normal(0, 0.8, len(actions))).clip(0)
    wns = baseline_chip[1] - 0.11 * peak - 0.0008 * changed + rng.normal(0, 0.0015, len(actions))
    tns = baseline_chip[2] - 0.55 * stress - 0.015 * changed + rng.normal(0, 0.025, len(actions))
    return baseline_chip, np.column_stack([drc, wns, tns]).astype(np.float32)


def write_csv(path: Path, fieldnames: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def checksums(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "predictor_baseline.npz"
    if args.reuse_baseline and cache_path.is_file():
        with np.load(cache_path, allow_pickle=True) as archive:
            baseline_state = {key: archive[key] for key in archive.files}
    else:
        baseline_state = infer_baseline_layer_state(args)
        np.savez_compressed(cache_path, **baseline_state)

    actions, family = create_assignments(args.seed)
    split = create_splits(family, args.seed)
    supply = pd.read_csv(args.baseline_root / "partition_layer_supply.csv").sort_values("r_index").reset_index(drop=True)
    supply.attrs["baseline_root"] = args.baseline_root
    adjacency = partition_adjacency(args.baseline_root)
    mean_after, p90_after, y_partition, baseline_partition = synthesize_utilization(
        actions, baseline_state["mean_pl"], baseline_state["p90_pl"], supply, adjacency
    )
    width_ratio = WIDTH_RATIO[actions]
    spacing_ratio = SPACING_RATIO[actions]
    sample_ids = np.asarray([f"S{i:04d}" for i in range(len(actions))])
    np.savez_compressed(
        args.output_dir / "samples.npz",
        sample_id=sample_ids,
        split=split,
        family=family,
        action_id=actions,
        width_ratio=width_ratio,
        spacing_ratio=spacing_ratio,
        utilization_mean_pl=mean_after,
        utilization_p90_pl=p90_after,
        utilization_mean_r=mean_after.reshape(len(actions), 60),
        utilization_p90_r=p90_after.reshape(len(actions), 60),
        y_partition_utilization=y_partition,
    )

    static_inputs = create_static_inputs(args, baseline_state, baseline_partition, args.seed)
    np.savez_compressed(args.output_dir / "static_inputs.npz", **static_inputs)
    critical = create_critical_nets(args, args.seed)
    np.savez_compressed(args.output_dir / "synthetic_timing_inputs.npz", **critical)
    baseline_chip, aux_chip = create_aux_targets(actions, y_partition, baseline_partition, args.seed)
    np.savez_compressed(
        args.output_dir / "synthetic_aux_targets.npz",
        baseline_chip_drc_wns_tns=baseline_chip,
        y_chip_drc_wns_tns_proxy=aux_chip,
    )

    manifest_rows = []
    for i in range(len(actions)):
        manifest_rows.append(
            {
                "sample_id": sample_ids[i],
                "split": split[i],
                "family": family[i],
                "num_changed": int(np.count_nonzero(actions[i])),
                "count_1W2S": int(np.count_nonzero(actions[i] == 1)),
                "count_1W3S": int(np.count_nonzero(actions[i] == 2)),
                "count_2W3S": int(np.count_nonzero(actions[i] == 3)),
            }
        )
    write_csv(args.output_dir / "sample_manifest.csv", list(manifest_rows[0]), manifest_rows)

    baseline_rows = []
    for row in supply.itertuples(index=False):
        p, layer = int(row.partition_index), int(row.layer_index)
        baseline_rows.append(
            {
                "r_index": int(row.r_index),
                "partition_id": row.partition_id,
                "layer_name": row.layer_name,
                "preferred_direction": row.preferred_direction,
                "congestion_mean_dir": float(baseline_state["mean_pl"][p, layer]),
                "congestion_p90_dir": float(baseline_state["p90_pl"][p, layer]),
                "direct_cell_aggregation": bool(baseline_state["direct_pl"][p, layer]),
                "cell_count": int(baseline_state["cell_counts_pl"][p, layer]),
            }
        )
    write_csv(args.output_dir / "partition_layer_baseline.csv", list(baseline_rows[0]), baseline_rows)

    count_distribution = Counter(np.count_nonzero(actions, axis=1).tolist())
    split_distribution = Counter(split.tolist())
    validation = {
        "status": "pass",
        "num_samples": int(len(actions)),
        "action_shape": list(actions.shape),
        "y_partition_utilization_shape": list(y_partition.shape),
        "utilization_mean_pl_shape": list(mean_after.shape),
        "utilization_p90_pl_shape": list(p90_after.shape),
        "num_changed_distribution": {str(k): int(v) for k, v in sorted(count_distribution.items())},
        "requested_buckets": {
            "one": int(np.sum(np.count_nonzero(actions, axis=1) == 1)),
            "two": int(np.sum(np.count_nonzero(actions, axis=1) == 2)),
            "three": int(np.sum(np.count_nonzero(actions, axis=1) == 3)),
            "four_to_ten": int(np.sum(np.count_nonzero(actions, axis=1) >= 4)),
        },
        "split_distribution": dict(split_distribution),
        "single_site_complete_coverage": bool(
            all(np.sum((actions[:, r] == action) & (np.count_nonzero(actions, axis=1) == 1)) == 1 for r in range(60) for action in (1, 2, 3))
        ),
        "finite": bool(
            np.isfinite(mean_after).all() and np.isfinite(p90_after).all() and np.isfinite(y_partition).all()
        ),
        "p90_ge_mean": bool(np.all(p90_after >= mean_after)),
        "baseline_action_exact": bool(np.all((actions == 0) == ((width_ratio == 1) & (spacing_ratio == 1)))),
        "r10_predictor_aggregation": "nearest 64 predicted cells; flagged direct_cell_aggregation=false",
    }
    if validation["requested_buckets"] != {"one": 180, "two": 90, "three": 90, "four_to_ten": 360}:
        raise AssertionError(validation["requested_buckets"])
    if dict(split_distribution) != {"train": 576, "validation": 72, "test": 72}:
        raise AssertionError(split_distribution)
    (args.output_dir / "validation_report.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    metadata = {
        "dataset_name": "phyndr_ndr_720_v1",
        "case": CASE,
        "seed": args.seed,
        "primary_target": "absolute partition H/V utilization, shape [720,10,2]",
        "target_semantics": "utilization; not delta-congestion and not capacity loss",
        "candidate_actions": {str(i): str(name) for i, name in enumerate(ACTION_NAMES)},
        "baseline_layer_state": {
            "shape": [10, 6, 2],
            "columns": ["congestion_mean_dir", "congestion_p90_dir"],
            "predictor": "RouteGNN_v4 UNet+Blockage-Encoder best checkpoint",
            "inference": "one physical routing layer retained at a time",
            "placement_macro_context": "frozen UNet gcell features (macro_region, RUDY, RUDY_pin) plus RouteGNN cell/net graph",
            "layer_blockage_context": ["routing_blockage", "pg_vdd", "pg_vss"],
            "aggregation": "raw utilization=expm1(model log output); mean/P90 over cells in each partition",
        },
        "synthetic_ndr_response": {
            "status": "proxy only; replace with registered OpenROAD experiment labels later",
            "utilization_gain_by_action": UTILIZATION_GAIN.tolist(),
            "capacity_loss_present": False,
            "effects": ["local action", "same-layer adjacent-partition spill", "adjacent-layer spill", "same-partition multi-action synergy"],
        },
        "synthetic_auxiliary_data": {
            "power_proxy_u": "derived reproducibly from real cell/macro area and count",
            "critical_timing": "256 real high-score nets with synthetic slack/criticality/path membership",
            "chip_drc_wns_tns": "synthetic proxy, stored separately and not a primary label",
        },
        "files": {
            "samples.npz": "720 assignments and absolute utilization targets",
            "static_inputs.npz": "real/static graph-node inputs plus predictor baseline state and power proxy",
            "predictor_baseline.npz": "raw per-partition/per-layer predictor aggregation",
            "synthetic_timing_inputs.npz": "critical-net proxy features on real incidence topology",
            "synthetic_aux_targets.npz": "separate proxy DRC/WNS/TNS targets",
        },
        "source_paths": {
            "baseline_root": str(args.baseline_root.resolve()),
            "routegnn_checkpoint": str(args.routegnn_checkpoint.resolve()),
            "blockage_checkpoint": str(args.blockage_checkpoint.resolve()),
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = "# PhyNDR synthetic NDR dataset (720 samples)\n\n"
    readme += "Primary labels are absolute H/V utilization in `samples.npz`; no capacity-loss field exists. "
    readme += "The 60 baseline `(partition, layer)` mean/P90 features are genuine predictions from the frozen RouteGNN_v4 UNet+Blockage-Encoder chain. "
    readme += "The action response, timing, power, DRC, WNS and TNS values are deterministic synthetic proxies and must be replaced by OpenROAD measurements before final training/evaluation.\n\n"
    readme += "Assignment counts: 180 one-site (complete 60x3 coverage), 90 two-site, 90 three-site, and 360 four-to-ten-site samples.\n"
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    checksums(args.output_dir)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
