from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class FrozenConfig:
    embedding_dimension: int = 1024
    segment_max_length: int = 1024
    segment_overlap_tokens: int = 128
    segmentation: str = "qwen3-token-offset-v2"
    segment_encoding_format: str = "title-section-passage-v1"
    query_max_length: int = 512
    entity_max_length: int = 64
    segment_roots: int = 64
    entity_roots_per_mention: int = 4
    entity_roots_cap: int = 16
    entity_similarity_min: float = 0.55
    softmax_temperature: float = 0.05
    entity_seed_mass: float = 0.5
    restart: float = 0.15
    epsilon: float = 1e-5
    max_pushes: int = 1_000_000
    hnsw_m: int = 32
    hnsw_ef_construction: int = 128
    # Included in the persisted configuration hash.
    hnsw_ef_search: int = 128
    @property
    def sha256(self) -> str:
        identity = {
            "parameters": asdict(self),
            "entity_labels": ENTITY_LABELS,
            "segment_query_instruction": SEGMENT_QUERY_INSTRUCTION,
            "entity_query_instruction": ENTITY_QUERY_INSTRUCTION,
        }
        value = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


DEFAULT_HNSW_EF_SEARCH = 512


ENTITY_LABELS = (
    "person",
    "organization",
    "location",
    "creative work",
    "event",
    "award",
    "profession or role",
    "disease or medical condition",
)

SEGMENT_QUERY_INSTRUCTION = (
    "Given a web research question, retrieve passages that provide evidence "
    "needed to answer the question"
)
ENTITY_QUERY_INSTRUCTION = (
    "Retrieve the same named entity or a lexical alias of the named mention"
)
