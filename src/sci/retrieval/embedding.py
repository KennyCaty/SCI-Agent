from pathlib import Path
from typing import Sequence
import numpy as np
import time
import os
from sci.retrieval.asset_config import SEGMENT_QUERY_INSTRUCTION
from sci.retrieval.cuda_runtime import DEVICE,record_call

def format_query(query,instruction):
    return f"Instruct: {instruction.strip()}\nQuery: {query.strip()}"

class QueryEncoder:
    def __init__(
        self,
        model_path: Path,
        dimension: int,
        device: str = "cuda",
        attention_implementation: str = "sdpa",
        ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
    
        self.dimension = int(dimension)
        self.query_max_length = int(os.environ.get("CEP_QUERY_MAX_LENGTH", "512"))
        self.device = DEVICE
        if str(torch.device(device)) not in ("cuda", str(DEVICE)):
            raise ValueError("Encoder device differs from the configured CUDA device")
        self.attention_implementation = str(attention_implementation)
        if self.device.type != "cuda" and self.attention_implementation != "sdpa":
            raise ValueError("FlashAttention requires a CUDA device")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, padding_side="left"
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=self.attention_implementation,
        ).to(self.device)
        self.model.eval()
        self.model.config.use_cache = False
        if self.dimension > int(self.model.config.hidden_size):
            raise ValueError("embedding dimension exceeds model hidden size")

    def encode(self, texts: Sequence[str], max_length: int) -> np.ndarray:
        import torch
        import torch.nn.functional as functional
    
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float16)
        begin = time.perf_counter()
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            left_padding = bool(batch["attention_mask"][:, -1].sum() == hidden.shape[0])
            if left_padding:
                pooled = hidden[:, -1]
            else:
                lengths = batch["attention_mask"].sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=self.device), lengths]
            vectors = functional.normalize(pooled[:, : self.dimension].float(), p=2, dim=1)
        result = vectors.cpu().numpy().astype(np.float16)
        record_call("encode", begin, rows=len(texts), dimension=self.dimension, input_kind="query")
        return result

    def segment_queries(self, queries: Sequence[str], max_length: int) -> np.ndarray:
        return self.encode(
            [format_query(query, SEGMENT_QUERY_INSTRUCTION) for query in queries], self.query_max_length
        )
