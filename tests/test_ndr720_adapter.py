from pathlib import Path
import pytest

from phyndr.data.graph_builder import validate_graph
from phyndr.data.ndr720 import NDR720Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE = "zero-riscy_freq_200_mp_4_fpu_70_fpa_1.5_p_4_fi_ap"


def test_ndr720_adapter_contract():
    dataset_root = PROJECT_ROOT / "datasets" / "phyndr_ndr_720_v1"
    if not dataset_root.is_dir():
        pytest.skip("optional phyndr_ndr_720_v1 dataset is not installed")
    dataset = NDR720Dataset(
        PROJECT_ROOT / "datasets" / "phyndr_ndr_720_v1",
        PROJECT_ROOT.parent / "baseline" / CASE / "phyndr_available_inputs_v1",
        "validation",
    )
    graph, target, baseline, _ = dataset[0]
    validate_graph(graph)
    assert len(dataset) == 72
    assert graph.num_nodes("r") == 60
    assert graph.num_nodes("u") == 10
    assert graph.nodes["u"].data["x_state"].shape == (10, 12)
    assert target.shape == baseline.shape == (10, 2)
    assert "utilization_mean_pl" not in graph.nodes["r"].data
