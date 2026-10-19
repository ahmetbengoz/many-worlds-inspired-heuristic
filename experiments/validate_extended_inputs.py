"""Cross-check the added TSPLIB matrices against the independent tsplib95 reader."""
from pathlib import Path
import sys, json
import numpy as np
import tsplib95

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tsplib_extended import parse_extended


def main():
    manifest = json.loads((ROOT / 'results/v6/dataset_manifest.json').read_text())
    rows = []
    for item in manifest:
        if item['original'] or item['suite'] != 'TSPLIB':
            continue
        path = ROOT / item['path']
        matrix = parse_extended(path).dist
        independent = tsplib95.load(str(path))
        nodes = list(independent.get_nodes())
        rng = np.random.default_rng(95371)
        checked = 0
        for i, j in rng.integers(0, len(nodes), size=(1000, 2)):
            if i == j:
                continue
            assert int(matrix[i, j]) == independent.get_weight(nodes[i], nodes[j]), (
                item['instance'], int(i), int(j))
            checked += 1
        rows.append(dict(instance=item['instance'], kind=item['edge_weight_type'],
                         pair_draws=1000, nonself_pairs_checked=checked,
                         independent_reader='tsplib95 0.7.1', seed=95371))
    assert len(rows) == 12
    (ROOT / 'results/v6/distance_validation.json').write_text(json.dumps(rows, indent=2))
    print('Validated 12 added TSPLIB matrices on', sum(x['nonself_pairs_checked'] for x in rows),
          'non-self sampled pairs')


if __name__ == '__main__':
    main()
