import numpy as np
import pytest
import tsplib95
from src.tsplib_extended import parse_extended


@pytest.mark.parametrize('storage,weights', [
    ('FULL_MATRIX', '0 2 3 2 0 5 3 5 0'),
    ('UPPER_ROW', '2 3 5'),
    ('UPPER_DIAG_ROW', '0 2 3 0 5 0'),
    ('LOWER_ROW', '2 3 5'),
    ('LOWER_DIAG_ROW', '0 2 0 3 5 0'),
])
def test_explicit_storage_is_distance_preserving(tmp_path, storage, weights):
    path = tmp_path / 'three.tsp'
    path.write_text('NAME : three\nTYPE : TSP\nDIMENSION : 3\n'
                    'EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : ' + storage +
                    '\nEDGE_WEIGHT_SECTION\n' + weights + '\nEOF\n')
    assert np.array_equal(parse_extended(path).dist, [[0, 2, 3], [2, 0, 5], [3, 5, 0]])


@pytest.mark.parametrize('kind,coordinates', [
    ('ATT', '1 0 0\n2 3 4\n3 10 0'),
    ('GEO', '1 38.24 20.42\n2 39.57 26.15\n3 40.56 25.32'),
])
def test_coordinate_distances_match_independent_reader(tmp_path, kind, coordinates):
    path = tmp_path / 'coordinates.tsp'
    path.write_text('NAME : coordinates\nTYPE : TSP\nDIMENSION : 3\n'
                    'EDGE_WEIGHT_TYPE : ' + kind + '\nNODE_COORD_SECTION\n' +
                    coordinates + '\nEOF\n')
    matrix = parse_extended(path).dist
    independent = tsplib95.load(str(path))
    for i in range(3):
        for j in range(3):
            if i != j:
                assert matrix[i, j] == independent.get_weight(i + 1, j + 1)
    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0)
