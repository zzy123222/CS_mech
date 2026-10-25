from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.generate_graph_reasoning import generate as graph
from src.datasets.generate_error_monitor import generate as errors
from src.datasets.generate_hallucination import generate as hallucination
from src.metrics.behavior import pleiotropy_index
from src.utils.io import save_table, load_table


def main():
    g = graph(8, seed=1)
    e = errors(8, seed=11, n_steps=4)
    h = hallucination(8, seed=71, distractors=4)
    assert len(g) == 32 and len(e) == 8 and len(h) == 32
    assert 0 < pleiotropy_index([1, 2, 3]) <= 1
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.csv"
        save_table(g, p)
        g2 = load_table(p)
        assert len(g2) == len(g)
    print("offline smoke test passed")


if __name__ == "__main__":
    main()
