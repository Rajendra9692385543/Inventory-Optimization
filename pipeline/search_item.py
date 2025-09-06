# pipeline/search_item.py
from ultralytics import YOLO
from PIL import Image
import numpy as np
from embeddings.embedding_model import ResNetEmbedder
from embeddings.indexer import FaissIndexer
from database.db_helper import get_items_by_faiss_ids, get_item_by_id, init_db
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = DATA_ROOT / "embeddings" / "index.faiss"
EMBED_DIM = 2048

def search_image(image_path: str,
                 detector_model: str = None,
                 top_k_per_crop: int = 5,
                 global_top_k: int = 5,
                 conf_threshold: float = 0.25,
                 match_threshold: float = 0.8):
    """
    Search pipeline: detect crops -> embed -> FAISS search -> map to inventory items.
    Returns a single dict {faiss_id, score, item_id, item_name, sku}
    if top match passes match_threshold; otherwise returns a message string.
    """
    init_db()
    detector = YOLO(detector_model if detector_model else "yolov8n.pt")
    embedder = ResNetEmbedder()
    indexer = FaissIndexer(dim=EMBED_DIM, index_path=str(FAISS_INDEX_PATH))

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Query image not found: {image_path}")

    # --- 1. Detect objects in the query image ---
    results = detector.predict(source=image_path, conf=conf_threshold, verbose=False)
    crops = []
    if len(results) == 0:
        crops = [Image.open(image_path).convert("RGB")]
    else:
        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            for xy in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, xy)
                crop = Image.open(image_path).convert("RGB").crop((x1, y1, x2, y2))
                crops.append(crop)
        else:
            crops = [Image.open(image_path).convert("RGB")]

    # --- 2. Embed the crops ---
    query_vectors = embedder.embed_batch(crops).astype('float32')
    D, I = indexer.search(query_vectors, top_k=top_k_per_crop)  # shapes (Q, K)

    # --- 3. Collect hits ---
    hits = []
    for q_idx in range(I.shape[0]):
        for k_idx in range(I.shape[1]):
            faiss_id = int(I[q_idx, k_idx])
            score = float(D[q_idx, k_idx])  # inner-product ~ cosine
            hits.append((faiss_id, score))

    # sort by score and take unique top results
    hits_sorted = sorted(hits, key=lambda x: x[1], reverse=True)
    seen = set()
    top_hits = []
    for faiss_id, score in hits_sorted:
        if faiss_id in seen or faiss_id < 0:
            continue
        seen.add(faiss_id)
        top_hits.append((faiss_id, score))
        if len(top_hits) >= global_top_k:
            break

    if not top_hits:
        return "Product not found in inventory"

    # --- 4. Map FAISS IDs back to item info ---
    faiss_ids = [h[0] for h in top_hits]
    db_rows = get_items_by_faiss_ids(faiss_ids)
    mapping = {r[0]: r[1] for r in db_rows}  # faiss_id -> item_id

    results_out = []
    for faiss_id, score in top_hits:
        item_id = mapping.get(faiss_id)
        if item_id is None:
            continue
        item_row = get_item_by_id(item_id)
        if not item_row:
            continue
        results_out.append({
            'faiss_id': faiss_id,
            'score': score,
            'item_id': item_row[0],
            'item_name': item_row[1],
            'sku': item_row[2],
        })

    # --- 5. Keep only best match if above threshold ---
    if not results_out:
        return "Product not found in inventory"

    best_match = results_out[0]
    if best_match['score'] >= match_threshold:
        return best_match
    else:
        return "Product not found in inventory"

if __name__ == "__main__":
    result = search_image("queries/test_1.jpg")
    print(result)
