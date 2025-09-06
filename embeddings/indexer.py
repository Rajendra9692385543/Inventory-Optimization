# embeddings/indexer.py
import faiss
import numpy as np
from pathlib import Path

class FaissIndexer:
    """
    Wraps a FAISS IndexFlatIP. Assumes vectors are L2-normalized float32.
    """
    def __init__(self, dim: int, index_path: str = "embeddings/index.faiss"):
        self.dim = dim
        self.index_path = Path(index_path)
        self.index = None
        self._init_index()

    def _init_index(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    def add(self, vectors: np.ndarray) -> None:
        assert vectors.dtype == np.float32
        assert vectors.shape[1] == self.dim
        self.index.add(vectors)

    def search(self, queries: np.ndarray, top_k: int = 5):
        D, I = self.index.search(queries, top_k)
        return D, I

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
