"""Minimal TSPLIB parser for EUC_2D TSP instances."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np


@dataclass(frozen=True)
class TSPInstance:
    name: str
    dimension: int
    edge_weight_type: str
    coords: np.ndarray
    dist: np.ndarray


def _split_header(line: str):
    if ':' in line:
        k, v = line.split(':', 1)
    else:
        parts = line.split(maxsplit=1)
        if len(parts) == 1:
            return parts[0].strip(), ''
        k, v = parts
    return k.strip().upper(), v.strip()


def tsplib_round(x: float) -> int:
    """TSPLIB EUC_2D integer rounding: int(distance + 0.5)."""
    return int(x + 0.5)


def parse_tsp(path: str | Path) -> TSPInstance:
    path = Path(path)
    lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    header = {}
    coords = []
    in_coords = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.upper() == 'NODE_COORD_SECTION':
            in_coords = True
            continue
        if line.upper() == 'EOF':
            break
        if not in_coords:
            k, v = _split_header(line)
            header[k] = v
        else:
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
    name = header.get('NAME', path.stem)
    dim = int(header.get('DIMENSION', len(coords)))
    ewt = header.get('EDGE_WEIGHT_TYPE', 'EUC_2D').upper()
    if ewt != 'EUC_2D':
        raise NotImplementedError(f'{path.name}: only EUC_2D is currently supported, got {ewt}')
    if len(coords) != dim:
        raise ValueError(f'{path.name}: expected {dim} coordinates, found {len(coords)}')
    xy = np.asarray(coords, dtype=float)
    diff = xy[:, None, :] - xy[None, :, :]
    dist_float = np.sqrt(np.sum(diff * diff, axis=2))
    dist = np.floor(dist_float + 0.5).astype(np.int32)
    np.fill_diagonal(dist, 0)
    return TSPInstance(name=name, dimension=dim, edge_weight_type=ewt, coords=xy, dist=dist)


def load_directory(data_dir: str | Path) -> list[TSPInstance]:
    return [parse_tsp(p) for p in sorted(Path(data_dir).glob('*.tsp'))]
