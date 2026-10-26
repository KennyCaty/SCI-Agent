import json
import os
from pathlib import Path
import sys

from sci.paths import ROOT
PROJECT = ROOT


def configure_embedding(options=None):
    options = options or {}
    profiles = json.loads((ROOT / 'configs/retrieval/embedding_profiles.json').read_text())
    name = str(options.get('profile', '256'))
    if name not in profiles:
        raise ValueError(f'Unknown embedding profile {name!r}; choose 256 or 1024')
    profile = {**profiles[name], **options}
    dimension = profiles[name]['dimension']
    if profile.get('dimension') != dimension:
        raise ValueError('Embedding profile and dimension disagree')
    if profile.get('backend', 'cuda') != 'cuda':
        raise ValueError('This implementation requires the CUDA backend')
    if profile.get('background_backend', 'cuda') != 'cuda':
        raise ValueError('This implementation uses CUDA background computation')
    scale = str(profile.get('corpus_scale', '100k'))
    if scale not in ('100k', '250k', '500k', '1m'):
        raise ValueError('Unknown corpus scale')
    resources = profile.get('resources_config') or os.environ.get('CEP_RESOURCES_CONFIG') or ROOT / 'configs/retrieval' / ('browsecomp_plus.json' if scale == '100k' else f'{scale}.json')
    os.environ['CEP_RESOURCES_CONFIG'] = str(Path(resources).expanduser().resolve())
    os.environ['CEP_CUDA_DEVICE'] = str(profile.get('device', os.environ.get('CEP_CUDA_DEVICE', 'cuda:0')))
    query_max_length = profile.get('query_max_length', 512)
    if query_max_length not in (512, 8192):
        raise ValueError('Supported query_max_length values are 512 and 8192')
    os.environ['CEP_EMBEDDING_DIMENSION'] = str(dimension)
    os.environ['CEP_QUERY_MAX_LENGTH'] = str(query_max_length)
    for key in ('graph_assets', 'embedding_model', 'corpus', 'bm25_index'):
        if profile.get(key):
            p = Path(profile[key]).expanduser()
            os.environ['CEP_' + key.upper()] = str((PROJECT / p).resolve() if not p.is_absolute() else p.resolve())
    from sci.retrieval.paths import resource
    from sci.retrieval.asset_config import FrozenConfig
    selected = FrozenConfig(embedding_dimension=dimension, query_max_length=query_max_length)
    assets = resource('graph_assets')
    ready = assets.parent / 'READY.json'
    if not ready.is_file() and os.environ.get('SCI_INDEX_VALIDATING') != '1':
        raise RuntimeError('SCI index is not READY; inspect outputs/indexing/logs/BUILD_STATUS.json and outputs/indexing/logs/BUILD_ERROR.json')
    dense_path = assets / ('segment_dense' if dimension == 256 else 'segment_dense1024') / 'manifest.json'
    graph_path = assets / 'graph/manifest.json'
    if not dense_path.is_file() or not graph_path.is_file():
        raise FileNotFoundError(f'Embedding {name} assets are not available: {assets}')
    dense = json.loads(dense_path.read_text())
    graph = json.loads(graph_path.read_text())
    if dense.get('state') != 'complete' or dense.get('kind') != 'segment':
        raise ValueError('Selected dense assets are not a complete segment index')
    if dense.get('dimension') != dimension:
        raise ValueError(f"Index dimension {dense.get('dimension')} differs from selected {dimension}")
    for label, manifest in [('dense', dense), ('graph', graph)]:
        if manifest.get('config_sha256_by_dimension', {}).get(str(dimension), manifest.get('config_sha256')) != selected.sha256:
            raise ValueError(f'{label} configuration hash differs from selected embedding profile')
    rows, capacity = int(dense['row_count']), int(dense['capacity'])
    if not 0 < rows <= capacity:
        raise ValueError('Invalid dense row count/capacity')
    if profile.get('expected_source_map'):
        expected = json.loads(Path(profile['expected_source_map']).read_text())
        actual = json.loads(Path(profile['source_map']).read_text())
        if expected != actual:
            raise ValueError('Source-ID mapping differs from evaluation mapping')
    dense_dir = dense_path.parent
    for field, expected in [('embedding_file', capacity * dimension * 2), ('id_file', capacity * 8)]:
        path = (dense_dir / dense[field]).resolve()
        if path.parent != dense_dir.resolve() or path.stat().st_size != expected:
            raise ValueError(f'{field} size/path differs from selected index shape')
    return dict(backend="cuda", corpus_scale=scale, profile=name, dimension=dimension, query_max_length=query_max_length, graph_assets=str(assets),
                embedding_model=str(resource('embedding_model')), config_sha256=selected.sha256,
                row_count=rows, manifest_path=str(dense_path),
                **({'expected_source_map': profile['expected_source_map']} if profile.get('expected_source_map') else {}))
