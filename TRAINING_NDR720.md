# PhyNDR 720-sample training contract

This document supersedes the old `delta_congestion` and capacity-scaling text
in the original v0.3.3 README for the current dataset adapter.

## Current primary task

- Input: one pre-NDR baseline design plus one 60-node `R(P,L)` action assignment.
- Actions: `0=1W1S`, `1=1W2S`, `2=1W3S`, `3=2W3S`.
- Primary output: `partition_utilization [N_P,2]` in `[H,V]` order.
- Label: absolute utilization, not delta-congestion and not capacity loss.
- `1W1S` has zero direct action pressure and is equivalent to no registered region NDR.

The model preserves baseline physical supply. Width/spacing ratios enter the
action encoder, but are not converted to a hand-written capacity reduction.
The partition head learns a residual on top of the explicit
`baseline_utilization [N_P,2]` input and returns the absolute utilization.

## Adapter

`src/phyndr/data/ndr720.py` maps the archives to a fixed heterogeneous graph:

- 60 `r=R(P,L)` nodes, 10 `u=U(P)` nodes, 256 critical-net nodes;
- same-layer H/V edges, cross-layer edges, ownership edges, boundary edges,
  and real partition/net incidence edges;
- `r.x_state` uses predictor-derived per-layer mean/P90;
- `u.x_state` contains 8 physical/net fields plus 4 PowerMap proxy fields;
- only baseline features are inputs; post-action utilization is label-only.

## Training

```powershell
$env:DGLBACKEND = "pytorch"
$env:PYTHONPATH = "$PWD\src"
python tools\train_ndr720.py `
  --output-dir experiments\ndr720\huber_rw1 `
  --loss huber --response-weight 1 --epochs 160
```

Model selection uses validation `active_response_mae`. Reports always compare
against the baseline-only predictor, because absolute MAE alone can hide a
model that ignores the action assignment.

## Initial experiments

| Method | Epochs | Test MAE | Baseline MAE | Improvement | Response R2 H/V |
|---|---:|---:|---:|---:|---:|
| Huber + response weighting | 40 | 0.000957 | 0.006684 | 85.7% | 0.980 / 0.985 |
| Huber, no response weighting | 30 | 0.001135 | 0.006684 | 83.0% | 0.966 / 0.980 |
| MSE + response weighting | 30 | 0.001518 | 0.006684 | 77.3% | 0.965 / 0.964 |

The preferred starting point is Huber with response weighting. These results
only establish that the architecture can learn the deterministic synthetic
action response. They do not establish generalization to OpenROAD labels or
to a different chip baseline.

## Next real-data step

Replace the proxy utilization targets while preserving stable `r` and `u`
indices. Keep the current split for pipeline regression tests, then create a
case-level split across multiple baseline designs for any scientific result.
