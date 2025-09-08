import os
import glob
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
import faiss
from supabase import create_client, Client

# ========== WINDOWS CONSOLE COMPATIBILITY ==========
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# ========== SETUP LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inventory_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
BASE_DIR = Path(__file__).resolve().parent
FAISS_INDEX_PATH = BASE_DIR / "faiss_index.bin"
TEMP_DIR = BASE_DIR / "temp"
UPLOADS_DIR = BASE_DIR / "uploads"
TEMP_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Constants
EMBED_DIM = 512
BUCKET_NAME = "inventory"
DEFAULT_DETECTOR_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_SIMILARITY_THRESHOLD = 0.5

# ========== ENVIRONMENT SETUP ==========
def load_environment():
    """Load environment variables from .env file or prompt user"""
    env_path = BASE_DIR / ".env"
    
    if env_path.exists():
        # Simple .env parser (avoiding python-dotenv dependency)
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    # Check if required variables exist
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        print("\nMissing Supabase credentials!")
        print("Please create a .env file with:")
        print("SUPABASE_URL=https://your-project.supabase.co")
        print("SUPABASE_ANON_KEY=your_anon_key_here")
        return False
    
    return True

# Load environment
if not load_environment():
    sys.exit(1)

# ========== SUPABASE SETUP ==========
def get_supabase_client() -> Client:
    """Initialize Supabase client with error handling"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    
    if not url or key is None:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_ANON_KEY environment variables")
    
    return create_client(url, key)

try:
    supabase = get_supabase_client()
    logger.info("SUCCESS: Supabase client initialized successfully")
except Exception as e:
    logger.error(f"ERROR: Failed to initialize Supabase: {e}")
    supabase = None

# ========== DATA MODELS ==========

class ProductInfo:
    """Data class for product information"""
    def __init__(self, name: str, sku: str, price: float, stock_quantity: int = 0, 
                 description: str = "", category: str = ""):
        self.name = name
        self.sku = sku
        self.price = float(price)
        self.stock_quantity = int(stock_quantity)
        self.description = description
        self.category = category
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for database insertion"""
        return {
            "name": self.name,
            "sku": self.sku,
            "price": self.price,
            "stock_quantity": self.stock_quantity,
            "description": self.description,
            "category": self.category
        }
    
    def __str__(self) -> str:
        return f"{self.name} (SKU: {self.sku}) - ${self.price:.2f} - Stock: {self.stock_quantity}"

# ========== CORE CLASSES ==========

class ResNetEmbedder:
    """ResNet-based image embedding generator"""
    
    def __init__(self, model_name: str = "resnet18"):
        logger.info(f"INIT: Initializing {model_name} embedder...")
        
        if model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif model_name == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model.fc = torch.nn.Identity()
        self.model = model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        logger.info(f"SUCCESS: ResNet embedder ready on {self.device}")
    
    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Generate embeddings for a batch of images"""
        if not images:
            return np.array([]).reshape(0, EMBED_DIM).astype(np.float32)
        
        tensors = [self.transform(img).unsqueeze(0) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(batch).cpu().numpy()
        
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.astype(np.float32)

class FaissIndexer:
    """FAISS-based similarity search index"""
    
    def __init__(self, dim: int = EMBED_DIM, index_path: str = str(FAISS_INDEX_PATH)):
        self.dim = dim
        self.index_path = index_path
        
        if os.path.exists(index_path):
            logger.info(f"LOAD: Loading existing FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
        else:
            logger.info("CREATE: Creating new FAISS index")
            base_index = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIDMap(base_index)
        
        logger.info(f"READY: FAISS index ready with {self.index.ntotal} vectors")
    
    def save(self) -> None:
        try:
            faiss.write_index(self.index, self.index_path)
            logger.debug(f"SAVE: Index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"ERROR: Failed to save index: {e}")
    
    def add(self, vectors: np.ndarray, ids: np.ndarray) -> np.ndarray:
        try:
            self.index.add_with_ids(vectors, ids.astype(np.int64))
            self.save()
            logger.debug(f"ADD: Added {len(vectors)} vectors to index")
            return ids
        except Exception as e:
            logger.error(f"ERROR: Failed to add vectors to index: {e}")
            raise
    
    def search(self, vectors: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        try:
            distances, ids = self.index.search(vectors, top_k)
            return distances, ids
        except Exception as e:
            logger.error(f"ERROR: Search failed: {e}")
            return np.array([]), np.array([])

class ImageProcessor:
    """Image preprocessing and object detection utilities"""
    
    def __init__(self, detector_model: str = DEFAULT_DETECTOR_MODEL):
        logger.info(f"INIT: Loading YOLO model: {detector_model}")
        self.detector = YOLO(detector_model)
        logger.info("SUCCESS: YOLO detector ready")
    
    def detect_and_crop(self, image_path: str, confidence: float = DEFAULT_CONFIDENCE) -> List[Image.Image]:
        try:
            results = self.detector.predict(source=image_path, conf=confidence, verbose=False)
            crops = []
            
            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
                img = Image.open(image_path).convert('RGB')
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    crop = img.crop((x1, y1, x2, y2))
                    crops.append(crop)
                logger.debug(f"DETECT: Found {len(crops)} objects in {image_path}")
            else:
                crops = [Image.open(image_path).convert('RGB')]
                logger.debug(f"DETECT: No objects detected, using full image: {image_path}")
            
            return crops
            
        except Exception as e:
            logger.error(f"ERROR: Detection failed for {image_path}: {e}")
            return [Image.open(image_path).convert('RGB')]

# ========== MAIN APPLICATION CLASS ==========

class InventoryManagementSystem:
    """Complete inventory management system"""
    
    def __init__(self, detector_model: str = DEFAULT_DETECTOR_MODEL, resnet_model: str = "resnet18"):
        logger.info("INIT: Initializing Inventory Management System...")
        
        if not supabase:
            raise RuntimeError("Supabase client not available")
        
        self.embedder = ResNetEmbedder(resnet_model)
        self.indexer = FaissIndexer()
        self.processor = ImageProcessor(detector_model)
        
        logger.info("SUCCESS: Inventory Management System ready!")
    
    def upload_product(self, product_info: ProductInfo, image_paths: List[str]) -> List[int]:
        """Upload a product with complete information and images"""
        logger.info(f"UPLOAD: Starting upload for product: {product_info.name}")
        created_ids = []
        
        for img_path in image_paths:
            if not Path(img_path).exists():
                logger.warning(f"WARNING: Image not found: {img_path}")
                continue
            
            try:
                # Insert product record
                product_data = product_info.to_dict()
                response = supabase.table("products").insert(product_data).execute()
                
                if not response.data:
                    logger.error("ERROR: Failed to insert product into database")
                    continue
                
                product_id = response.data[0]["id"]
                created_ids.append(product_id)
                
                # Process image and detect objects
                crops = self.processor.detect_and_crop(img_path)
                
                # Generate embeddings
                embeddings = self.embedder.embed_batch(crops)
                
                # Add to FAISS index
                ids = np.full((embeddings.shape[0],), product_id, dtype=np.int64)
                self.indexer.add(embeddings, ids)
                
                # Upload image to storage
                file_name = f"{product_id}/{Path(img_path).name}"
                with open(img_path, "rb") as f:
                    supabase.storage.from_(BUCKET_NAME).upload(file_name, f, {"upsert": True})
                
                # Get public URL and update record
                image_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
                
                supabase.table("products").update({
                    "image_url": image_url
                }).eq("id", product_id).execute()
                
                logger.info(f"SUCCESS: Uploaded '{product_info.name}' (ID: {product_id}) from {img_path}")
                
            except Exception as e:
                logger.error(f"ERROR: Failed to upload product from {img_path}: {e}")
                continue
        
        return created_ids
    
    def search_products(self, query_image_path: str, top_k: int = 5, 
                       name_filter: Optional[str] = None, category_filter: Optional[str] = None,
                       min_price: Optional[float] = None, max_price: Optional[float] = None,
                       in_stock_only: bool = False,
                       confidence_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> List[Dict]:
        """Search for products with detailed filtering options - Returns only highest similarity product"""
        if not Path(query_image_path).exists():
            logger.error(f"ERROR: Query image not found: {query_image_path}")
            return []
        
        try:
            logger.info(f"SEARCH: Searching for highest similarity product using: {query_image_path}")
            
            # Process query image
            crops = self.processor.detect_and_crop(query_image_path)
            embeddings = self.embedder.embed_batch(crops)
            
            # Search in FAISS index - get more candidates to find the best after filtering
            distances, ids = self.indexer.search(embeddings, top_k * 5)
            
            # Process results with filtering
            best_product = None
            highest_similarity = -1.0
            seen_ids = set()
            
            for distance, product_id in zip(distances.flatten(), ids.flatten()):
                if product_id == -1 or product_id in seen_ids:
                    continue
                
                similarity_score = float(distance)
                if similarity_score < confidence_threshold:
                    continue
                
                seen_ids.add(product_id)
                
                # Build query with filters
                query = supabase.table("products").select(
                    "id, name, sku, price, stock_quantity, description, category, image_url, created_at"
                ).eq("id", int(product_id))
                
                # Apply filters
                if name_filter:
                    query = query.ilike("name", f"%{name_filter}%")
                if category_filter:
                    query = query.ilike("category", f"%{category_filter}%")
                if min_price is not None:
                    query = query.gte("price", min_price)
                if max_price is not None:
                    query = query.lte("price", max_price)
                if in_stock_only:
                    query = query.gt("stock_quantity", 0)
                
                row = query.execute()
                
                if row.data and similarity_score > highest_similarity:
                    product_data = row.data[0]
                    product_data['similarity_score'] = similarity_score
                    product_data['formatted_price'] = f"${product_data['price']:.2f}"
                    product_data['stock_status'] = "In Stock" if product_data['stock_quantity'] > 0 else "Out of Stock"
                    
                    # Update best product if this one has higher similarity
                    highest_similarity = similarity_score
                    best_product = product_data
            
            # Return only the best match or empty list
            if best_product:
                logger.info(f"SEARCH: Found best matching product: {best_product['name']} (Score: {highest_similarity:.3f})")
                return [best_product]  # Return as list with single item
            else:
                logger.info("SEARCH: No matching products found")
                return []
                
        except Exception as e:
            logger.error(f"ERROR: Product search failed: {e}")
            return []
    
    def get_product_details(self, product_id: int) -> Optional[Dict]:
        """Get complete details of a specific product"""
        try:
            response = supabase.table("products").select("*").eq("id", product_id).execute()
            
            if response.data:
                product = response.data[0]
                product['formatted_price'] = f"${product['price']:.2f}"
                product['stock_status'] = "In Stock" if product['stock_quantity'] > 0 else "Out of Stock"
                return product
            return None
            
        except Exception as e:
            logger.error(f"ERROR: Failed to get product details: {e}")
            return None
    
    def update_stock(self, product_id: int, new_quantity: int) -> bool:
        """Update stock quantity for a product"""
        try:
            response = supabase.table("products").update({
                "stock_quantity": new_quantity
            }).eq("id", product_id).execute()
            
            if response.data:
                logger.info(f"SUCCESS: Updated stock for product {product_id} to {new_quantity}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"ERROR: Failed to update stock: {e}")
            return False
    
    def update_price(self, product_id: int, new_price: float) -> bool:
        """Update price for a product"""
        try:
            response = supabase.table("products").update({
                "price": new_price
            }).eq("id", product_id).execute()
            
            if response.data:
                logger.info(f"SUCCESS: Updated price for product {product_id} to ${new_price:.2f}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"ERROR: Failed to update price: {e}")
            return False
    
    def get_inventory_summary(self) -> Dict:
        """Get comprehensive inventory statistics"""
        try:
            # Basic counts
            total_products = supabase.table("products").select("id", count="exact").execute().count
            
            # Stock statistics
            stock_query = supabase.table("products").select("stock_quantity, price").execute()
            
            if stock_query.data:
                products = stock_query.data
                total_value = sum(p['price'] * p['stock_quantity'] for p in products)
                in_stock = sum(1 for p in products if p['stock_quantity'] > 0)
                out_of_stock = total_products - in_stock
                total_items = sum(p['stock_quantity'] for p in products)
            else:
                total_value = 0
                in_stock = 0
                out_of_stock = 0
                total_items = 0
            
            # FAISS stats
            faiss_vectors = self.indexer.index.ntotal
            
            return {
                'total_products': total_products,
                'in_stock_products': in_stock,
                'out_of_stock_products': out_of_stock,
                'total_inventory_items': total_items,
                'total_inventory_value': f"${total_value:.2f}",
                'faiss_vectors': faiss_vectors,
                'index_file_size_mb': round(os.path.getsize(FAISS_INDEX_PATH) / (1024 * 1024), 2) if os.path.exists(FAISS_INDEX_PATH) else 0
            }
            
        except Exception as e:
            logger.error(f"ERROR: Failed to get inventory summary: {e}")
            return {}
    
    def batch_add_from_directory(self, directory_path: str, batch_size: int = 10, 
                                name_pattern: str = "filename") -> List[int]:
        """Add all images from a directory"""
        logger.info(f"BATCH: Batch processing directory: {directory_path}")
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        all_images = []
        
        for ext in image_extensions:
            pattern = str(Path(directory_path) / ext)
            all_images.extend(glob.glob(pattern))
            all_images.extend(glob.glob(pattern.upper()))
        
        logger.info(f"FOUND: Found {len(all_images)} images to process")
        
        all_created_ids = []
        
        # Process in batches
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i+batch_size]
            logger.info(f"BATCH: Processing batch {i//batch_size + 1}/{(len(all_images)-1)//batch_size + 1}")
            
            for img_path in batch:
                try:
                    # Generate item name based on pattern
                    if name_pattern == "filename":
                        item_name = Path(img_path).stem.replace('_', ' ').replace('-', ' ').title()
                    elif name_pattern == "directory":
                        item_name = Path(directory_path).name.replace('_', ' ').replace('-', ' ').title()
                    else:
                        item_name = Path(img_path).stem
                    
                    # Generate SKU and price
                    sku = f"AUTO_{Path(img_path).stem.upper()}"
                    price = 10.0  # Default price
                    
                    # Create product info
                    product_info = ProductInfo(
                        name=item_name,
                        sku=sku,
                        price=price,
                        stock_quantity=1
                    )
                    
                    # Upload product
                    created_ids = self.upload_product(product_info, [img_path])
                    all_created_ids.extend(created_ids)
                    
                except Exception as e:
                    logger.error(f"ERROR: Failed to process {img_path}: {e}")
                    continue
        
        logger.info(f"SUCCESS: Batch processing complete. Created {len(all_created_ids)} items.")
        return all_created_ids
    
    def validate_setup(self) -> bool:
        """Validate system setup"""
        logger.info("VALIDATE: Validating system setup...")
        issues = []
        
        # Check environment variables
        if not os.getenv("SUPABASE_URL"):
            issues.append("ERROR: SUPABASE_URL environment variable not set")
        if not os.getenv("SUPABASE_ANON_KEY") and not os.getenv("SUPABASE_KEY"):
            issues.append("ERROR: SUPABASE_ANON_KEY or SUPABASE_KEY environment variable not set")
        
        # Test database connection
        try:
            supabase.table("products").select("id").limit(1).execute()
            logger.info("SUCCESS: Database connection successful")
        except Exception as e:
            issues.append(f"ERROR: Database connection failed: {e}")
        
        # Test storage bucket
        try:
            supabase.storage.from_(BUCKET_NAME).list()
            logger.info("SUCCESS: Storage bucket accessible")
        except Exception as e:
            issues.append(f"ERROR: Storage bucket '{BUCKET_NAME}' not accessible: {e}")
        
        if issues:
            logger.error("VALIDATION FAILED: Setup validation failed:")
            for issue in issues:
                logger.error(f"  {issue}")
            return False
        else:
            logger.info("SUCCESS: All systems validated successfully!")
            return True

# ========== INTERACTIVE INTERFACE ==========

class InventoryInterface:
    """Interactive command-line interface"""
    
    def __init__(self):
        self.system = InventoryManagementSystem()
    
    def upload_product(self):
        """Interactive product upload"""
        print("\n=== UPLOAD PRODUCT ===")
        
        # Get product information
        name = input("Product Name: ").strip()
        sku = input("SKU: ").strip()
        
        while True:
            try:
                price = float(input("Price ($): ").strip())
                break
            except ValueError:
                print("Please enter a valid price!")
        
        while True:
            try:
                stock = int(input("Stock Quantity: ").strip())
                break
            except ValueError:
                print("Please enter a valid stock quantity!")
        
        description = input("Description (optional): ").strip()
        category = input("Category (optional): ").strip()
        
        # Get image paths
        print("\nImage paths (enter one per line, press Enter twice when done):")
        image_paths = []
        while True:
            path = input("Image path: ").strip()
            if not path:
                break
            if Path(path).exists():
                image_paths.append(path)
                print(f"  Added: {path}")
            else:
                print(f"  File not found: {path}")
        
        if not image_paths:
            print("No valid images provided!")
            return
        
        # Create product info
        product_info = ProductInfo(
            name=name,
            sku=sku,
            price=price,
            stock_quantity=stock,
            description=description,
            category=category
        )
        
        # Upload to system
        try:
            product_ids = self.system.upload_product(product_info, image_paths)
            
            if product_ids:
                print(f"\nSUCCESS: Product uploaded with IDs: {product_ids}")
                
                # Show summary
                summary = self.system.get_inventory_summary()
                print(f"Total products in inventory: {summary.get('total_products', 0)}")
            else:
                print("\nFAILED: Could not upload product")
                
        except Exception as e:
            print(f"\nERROR: {e}")
    
    def search_products(self):
        """Interactive product search - Returns single best match"""
        print("\n=== SEARCH PRODUCTS ===")
        
        # Get query image
        query_image = input("Query image path: ").strip()
        if not Path(query_image).exists():
            print(f"Image not found: {query_image}")
            return
        
        # Get search parameters
        name_filter = input("Filter by name (optional): ").strip() or None
        category_filter = input("Filter by category (optional): ").strip() or None
        
        try:
            min_price = input("Minimum price (optional): ").strip()
            min_price = float(min_price) if min_price else None
        except ValueError:
            min_price = None
        
        try:
            max_price = input("Maximum price (optional): ").strip()
            max_price = float(max_price) if max_price else None
        except ValueError:
            max_price = None
        
        in_stock_only = input("In stock only? (y/n): ").strip().lower() == 'y'
        
        # Search (returns single best match)
        try:
            results = self.system.search_products(
                query_image,
                top_k=1,  # We only want the best match
                name_filter=name_filter,
                category_filter=category_filter,
                min_price=min_price,
                max_price=max_price,
                in_stock_only=in_stock_only,
                confidence_threshold=0.3
            )
            
            if results:
                product = results[0]  # Get the single best match
                print(f"\nBEST MATCHING PRODUCT:")
                print("=" * 80)
                print(f"Name: {product['name']}")
                print(f"SKU: {product['sku']}")
                print(f"Price: {product['formatted_price']}")
                print(f"Stock: {product['stock_quantity']} ({product['stock_status']})")
                print(f"Category: {product.get('category', 'N/A')}")
                print(f"Similarity Score: {product['similarity_score']:.3f}")
                if product.get('description'):
                    print(f"Description: {product['description']}")
                print(f"Image URL: {product.get('image_url', 'N/A')}")
                print("=" * 80)
                
                # Show confidence level
                similarity = product['similarity_score']
                if similarity > 0.9:
                    print("CONFIDENCE: EXCELLENT MATCH (90%+)")
                elif similarity > 0.7:
                    print("CONFIDENCE: GOOD MATCH (70-90%)")
                elif similarity > 0.5:
                    print("CONFIDENCE: FAIR MATCH (50-70%)")
                else:
                    print("CONFIDENCE: LOW MATCH (<50%)")
                    
            else:
                print("\nNo matching products found")
                print("Try:")
                print("   - Using a clearer image")
                print("   - Adjusting price filters")
                print("   - Removing category filters")
                
        except Exception as e:
            print(f"\nERROR: {e}")
    
    def view_inventory_summary(self):
        """Show inventory summary"""
        print("\n=== INVENTORY SUMMARY ===")
        
        try:
            summary = self.system.get_inventory_summary()
            
            print(f"Total Products: {summary.get('total_products', 0)}")
            print(f"In Stock: {summary.get('in_stock_products', 0)}")
            print(f"Out of Stock: {summary.get('out_of_stock_products', 0)}")
            print(f"Total Items: {summary.get('total_inventory_items', 0)}")
            print(f"Total Value: {summary.get('total_inventory_value', '$0.00')}")
            print(f"Vector Index Size: {summary.get('index_file_size_mb', 0)} MB")
            
        except Exception as e:
            print(f"\nERROR: {e}")
    
    def quick_upload(self):
        """Quick upload with minimal input"""
        print("\n=== QUICK UPLOAD ===")
        print("Example: Ultratech Cement, SKU123, 299.99, 100, uploads/u_1.jpg uploads/u_2.jpg")
        
        data = input("Enter: Name, SKU, Price, Stock, Image1 Image2 ...: ").strip()
        parts = data.split(',')
        
        if len(parts) < 4:
            print("Invalid format! Need: Name, SKU, Price, Stock, Images")
            return
        
        try:
            name = parts[0].strip()
            sku = parts[1].strip()
            price = float(parts[2].strip())
            stock = int(parts[3].strip())
            
            # Get images from remaining parts or separate input
            if len(parts) > 4:
                image_paths = [p.strip() for p in parts[4].split()]
            else:
                images_input = input("Image paths (space separated): ").strip()
                image_paths = images_input.split()
            
            # Validate images
            valid_paths = [p for p in image_paths if Path(p).exists()]
            
            if not valid_paths:
                print("No valid images found!")
                return
            
            # Create and upload
            product_info = ProductInfo(name=name, sku=sku, price=price, stock_quantity=stock)
            
            product_ids = self.system.upload_product(product_info, valid_paths)
            
            if product_ids:
                print(f"SUCCESS: Product '{name}' uploaded with IDs: {product_ids}")
                
                # Test search with uploaded image
                test_search = input("Test search with uploaded image? (y/n): ").strip().lower()
                if test_search == 'y' and valid_paths:
                    print(f"\nTesting search with: {valid_paths[0]}")
                    results = self.system.search_products(valid_paths[0], top_k=1, confidence_threshold=0.1)
                    if results:
                        best_match = results[0]
                        print(f"Best Match: {best_match['name']} (Score: {best_match['similarity_score']:.3f})")
                    else:
                        print("No matches found (this shouldn't happen with your own image)")
            else:
                print("FAILED: Could not upload product")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    def update_stock_price(self):
        """Update stock or price for existing products"""
        print("\n=== UPDATE PRODUCT ===")
        
        try:
            product_id = int(input("Product ID: ").strip())
            
            product = self.system.get_product_details(product_id)
            
            if not product:
                print(f"Product with ID {product_id} not found!")
                return
            
            print(f"\nCurrent Product Info:")
            print(f"Name: {product['name']}")
            print(f"SKU: {product['sku']}")
            print(f"Price: {product['formatted_price']}")
            print(f"Stock: {product['stock_quantity']}")
            
            # Update options
            print("\nWhat would you like to update?")
            print("1. Stock quantity")
            print("2. Price")
            print("3. Both")
            
            choice = input("Choice (1-3): ").strip()
            
            if choice in ['1', '3']:
                try:
                    new_stock = int(input("New stock quantity: ").strip())
                    success = self.system.update_stock(product_id, new_stock)
                    if success:
                        print(f"Stock updated to {new_stock}")
                    else:
                        print("Failed to update stock")
                except ValueError:
                    print("Invalid stock quantity")
            
            if choice in ['2', '3']:
                try:
                    new_price = float(input("New price: ").strip())
                    success = self.system.update_price(product_id, new_price)
                    if success:
                        print(f"Price updated to ${new_price:.2f}")
                    else:
                        print("Failed to update price")
                except ValueError:
                    print("Invalid price")
                    
        except ValueError:
            print("Invalid product ID")
        except Exception as e:
            print(f"ERROR: {e}")
    
    def batch_upload_directory(self):
        """Batch upload from directory"""
        print("\n=== BATCH UPLOAD FROM DIRECTORY ===")
        
        directory = input("Directory path: ").strip()
        if not Path(directory).exists():
            print(f"Directory not found: {directory}")
            return
        
        try:
            batch_size = int(input("Batch size (default 10): ").strip() or "10")
        except ValueError:
            batch_size = 10
        
        print("Name pattern options:")
        print("1. filename - Use image filename as product name")
        print("2. directory - Use directory name for all products")
        print("3. custom - Use image filename as-is")
        
        pattern_choice = input("Choose pattern (1-3): ").strip()
        pattern_map = {"1": "filename", "2": "directory", "3": "custom"}
        name_pattern = pattern_map.get(pattern_choice, "filename")
        
        try:
            created_ids = self.system.batch_add_from_directory(directory, batch_size, name_pattern)
            print(f"SUCCESS: Batch uploaded {len(created_ids)} products")
            
            # Show summary
            summary = self.system.get_inventory_summary()
            print(f"Total products now: {summary.get('total_products', 0)}")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    def main_menu(self):
        """Main menu interface"""
        print("""
INVENTORY MANAGEMENT SYSTEM
==============================

1. Upload Product (Detailed)
2. Quick Upload
3. Search Products by Image (Best Match Only)
4. View Inventory Summary
5. Update Stock/Price
6. Batch Upload Directory
7. Exit

""")
    
    def run(self):
        """Main application loop"""
        print("Starting Inventory Management System...")
        
        # Test connection first
        try:
            if not self.system.validate_setup():
                print("System setup failed! Please check your configuration.")
                return
        except Exception as e:
            print(f"Failed to initialize system: {e}")
            return
        
        while True:
            self.main_menu()
            choice = input("Choose an option (1-7): ").strip()
            
            if choice == '1':
                self.upload_product()
            elif choice == '2':
                self.quick_upload()
            elif choice == '3':
                self.search_products()
            elif choice == '4':
                self.view_inventory_summary()
            elif choice == '5':
                self.update_stock_price()
            elif choice == '6':
                self.batch_upload_directory()
            elif choice == '7':
                print("Goodbye!")
                break
            else:
                print("Invalid choice! Please select 1-7.")
            
            input("\nPress Enter to continue...")

# ========== UTILITY FUNCTIONS ==========

def create_database_schema():
    """Create the database schema"""
    schema_sql = """
-- Enhanced Inventory Database Schema
-- Run this in your Supabase SQL Editor

-- Create the products table with all required fields
CREATE TABLE IF NOT EXISTS products (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    sku TEXT UNIQUE NOT NULL,
    price DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    stock_quantity INTEGER NOT NULL DEFAULT 0,
    description TEXT DEFAULT '',
    category TEXT DEFAULT '',
    image_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_products_name ON products USING gin(to_tsvector('english', name));
CREATE INDEX IF NOT EXISTS idx_products_sku ON products (sku);
CREATE INDEX IF NOT EXISTS idx_products_category ON products (category);
CREATE INDEX IF NOT EXISTS idx_products_price ON products (price);
CREATE INDEX IF NOT EXISTS idx_products_stock ON products (stock_quantity);
CREATE INDEX IF NOT EXISTS idx_products_created_at ON products (created_at DESC);

-- Create a trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$ language 'plpgsql';

CREATE TRIGGER update_products_updated_at 
    BEFORE UPDATE ON products 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();
"""
    return schema_sql

def setup_project_structure():
    """Create basic project structure"""
    directories = [
        "uploads",
        "temp", 
        "backups"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"Created directory: {dir_name}/")
    
    # Create .env.example
    env_example = """SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here"""
    
    with open(".env.example", "w") as f:
        f.write(env_example)
    print("Created .env.example file")
    
    # Create .gitignore
    gitignore_content = """.env
*.log
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
temp/
faiss_index.bin
*.jpg
*.png
*.jpeg
.vscode/
.idea/
.DS_Store
Thumbs.db"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("Created .gitignore file")
    
    # Create requirements.txt
    requirements = """torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
faiss-cpu>=1.7.4
pillow>=10.0.0
numpy>=1.24.0
supabase>=2.0.0"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("Created requirements.txt file")
    
    print("\nSetup Instructions:")
    print("1. Copy .env.example to .env and add your Supabase credentials")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the database schema in your Supabase SQL editor")
    print("4. Create 'inventory' bucket in Supabase storage")
    print("5. Add test images to uploads/ directory")

def demo_system():
    """Run a demo of the system - Updated to show single best match"""
    print("INVENTORY SYSTEM DEMO")
    print("=" * 50)
    
    try:
        # Initialize system
        print("\n1. Initializing system...")
        system = InventoryManagementSystem()
        
        # Validate setup
        if not system.validate_setup():
            print("System validation failed!")
            return
        
        print("System ready!")
        
        # Show initial summary
        print("\n2. Initial inventory summary:")
        summary = system.get_inventory_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Demo upload if images exist
        sample_images = list(UPLOADS_DIR.glob("*.jpg")) + list(UPLOADS_DIR.glob("*.png"))
        
        if sample_images:
            print(f"\n3. Found {len(sample_images)} images for demo")
            
            # Upload first image as demo
            demo_image = str(sample_images[0])
            product_name = sample_images[0].stem.replace('_', ' ').title()
            
            demo_product = ProductInfo(
                name=product_name,
                sku=f"DEMO_{sample_images[0].stem.upper()}",
                price=99.99,
                stock_quantity=10,
                description=f"Demo product from {sample_images[0].name}",
                category="Demo Category"
            )
            
            print(f"   Uploading demo product: {product_name}")
            product_ids = system.upload_product(demo_product, [demo_image])
            
            if product_ids:
                print(f"   Demo product uploaded with ID: {product_ids[0]}")
                
                # Demo search - now returns single best match
                print(f"\n4. Searching for best match with same image...")
                results = system.search_products(demo_image, top_k=1, confidence_threshold=0.1)
                
                if results:
                    best_match = results[0]
                    print(f"   BEST MATCH FOUND:")
                    print(f"     Product: {best_match['name']}")
                    print(f"     Similarity Score: {best_match['similarity_score']:.3f}")
                    print(f"     Price: {best_match['formatted_price']}")
                    print(f"     Stock: {best_match['stock_quantity']}")
                    
                    # Confidence assessment
                    score = best_match['similarity_score']
                    if score > 0.95:
                        print("     CONFIDENCE: PERFECT MATCH!")
                    elif score > 0.8:
                        print("     CONFIDENCE: EXCELLENT MATCH")
                    else:
                        print("     CONFIDENCE: GOOD MATCH")
                
                # Show updated summary
                print(f"\n5. Updated inventory summary:")
                summary = system.get_inventory_summary()
                for key, value in summary.items():
                    print(f"   {key}: {value}")
            else:
                print("   Failed to upload demo product")
        else:
            print("\n3. No sample images found in uploads/ directory")
            print("   Add some .jpg or .png files to uploads/ directory for demo")
        
        print(f"\nDEMO COMPLETED!")
        print(f"\nKEY FEATURE: System now returns only the HIGHEST SIMILARITY product!")
        
    except Exception as e:
        print(f"\nDEMO FAILED: {e}")
        import traceback
        traceback.print_exc()

# ========== MAIN FUNCTION ==========

def main():
    """Main function with command line argument handling"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "setup":
            print("Setting up project structure...")
            setup_project_structure()
            print("\nDatabase Schema:")
            print(create_database_schema())
            
        elif command == "demo":
            demo_system()
            
        elif command == "schema":
            print("Database Schema:")
            print(create_database_schema())
            
        else:
            print("Available commands:")
            print("  python inventory_system.py setup  - Setup project structure")
            print("  python inventory_system.py demo   - Run demo")
            print("  python inventory_system.py schema - Show database schema")
            print("  python inventory_system.py        - Start interactive interface")
    else:
        # Start interactive interface
        interface = InventoryInterface()
        interface.run()

if __name__ == "__main__":
    main()