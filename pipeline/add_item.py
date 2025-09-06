# pipeline/add_item.py
import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import numpy as np

from embeddings.embedding_model import ResNetEmbedder
from embeddings.indexer import FaissIndexer
from database.db_helper import add_item_return_id, add_embedding_map, init_db

# Constants
DATA_ROOT = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = DATA_ROOT / "embeddings" / "index.faiss"
EMBED_DIM = 2048

def load_detector(model_path: str = None):
    """
    Load YOLO detector. If model_path is None, tries 'yolov8n.pt'.
    Make sure the ultralytics package can access or download the model.
    """
    if model_path is None:
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(model_path)
    return model

def crop_from_bbox(image_path: str, xyxy: list) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = map(int, xyxy)
    # Clip coordinates
    w, h = img.size
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))

def add_images_for_item(item_name: str, sku: str, image_paths: list, detector_model: str = None):
    """
    Main entry: given an item_name, sku, and list of image_paths (local), detect crops,
    compute embeddings, add to FAISS index and record mapping in DB.
    """
    init_db()
    timestamp = datetime.now().isoformat(timespec='seconds')
    item_id = add_item_return_id(item_name, sku, timestamp)
    if item_id is None:
        raise RuntimeError("Failed to create or fetch item id in DB.")
    detector = load_detector(detector_model)
    embedder = ResNetEmbedder()
    indexer = FaissIndexer(dim=EMBED_DIM, index_path=str(FAISS_INDEX_PATH))
    current_total = indexer.index.ntotal
    added = 0

    for img_path in image_paths:
        img_path = str(Path(img_path))
        if not Path(img_path).exists():
            print(f"Missing image {img_path}, skipping.")
            continue

        # run detector
        results = detector.predict(source=img_path, conf=0.25, verbose=False)
        crops = []
        if len(results) == 0:
            crops = [Image.open(img_path).convert("RGB")]
        else:
            res = results[0]
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy_arr = boxes.xyxy.cpu().numpy()
                for xy in xyxy_arr:
                    crop = crop_from_bbox(img_path, xy.tolist())
                    crops.append(crop)
            else:
                crops = [Image.open(img_path).convert("RGB")]

        if not crops:
            crops = [Image.open(img_path).convert("RGB")]

        # compute embeddings
        vectors = embedder.embed_batch(crops)  # shape (N, EMBED_DIM)
        # ensure dtype float32
        vectors = vectors.astype('float32')
        # add to faiss
        indexer.add(vectors)
        # record mapping: faiss ids are sequential starting at current_total
        for i in range(vectors.shape[0]):
            faiss_id = int(current_total + added + i)
            add_embedding_map(faiss_id, item_id, img_path, timestamp)
        added += vectors.shape[0]

    # persist index
    indexer.save()
    print(f"Added {added} embeddings for item_id={item_id}.")
