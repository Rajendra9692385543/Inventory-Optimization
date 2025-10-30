import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import plotly.express as px
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
import faiss
from supabase import create_client, Client
import io
import json
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import hashlib


# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="INVENTORY OPTIMIZATION",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== LOGGING & BASIC SETUP ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
FAISS_INDEX_PATH = BASE_DIR / "faiss_index_vgg16.bin"

# Constants
EMBED_DIM = 2048
BUCKET_NAME = "inventory"
DEFAULT_DETECTOR_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.40
DEFAULT_SIMILARITY_THRESHOLD = 0.6
LOW_STOCK_THRESHOLD = 10
CRITICAL_STOCK_THRESHOLD = 5
REORDER_MULTIPLIER = 1.5
CURRENCY_SYMBOL = "₹"


# ========== CUSTOM CSS FOR PROFESSIONAL LOOK ==========
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #ff9800;
        --danger-color: #d62728;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .status-critical { background-color: #fee; color: #c00; }
    .status-warning { background-color: #fff3cd; color: #856404; }
    .status-success { background-color: #d4edda; color: #155724; }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Info boxes */
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .info-box.info { background-color: #e7f3ff; border-color: #2196F3; }
    .info-box.success { background-color: #e8f5e9; border-color: #4caf50; }
    .info-box.warning { background-color: #fff8e1; border-color: #ff9800; }
    .info-box.error { background-color: #ffebee; border-color: #f44336; }
    </style>
    """, unsafe_allow_html=True)


# ========== AUTHENTICATION ==========
def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def check_authentication():
    """Check if user is authenticated"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        show_login_page()
        return False
    return True

def show_login_page():
    """Display professional login page (registration tab removed)"""
    st.markdown("""
    <div class="main-header">
        <h1> INVENTORY OPTIMIZATION </h1>
        <p>Next-Generation Intelligent Inventory Management</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🔑 Login")
        st.markdown("Please enter your credentials to continue")
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="user@example.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("🚀 Login", use_container_width=True, type="primary")
        if submit:
            if email and password:
                hashed_pw = hash_password(password)
                try:
                    response = supabase.table("users").select("*").eq("email", email).eq("password_hash", hashed_pw).execute()
                    if response.data and len(response.data) > 0:
                        st.session_state.authenticated = True
                        st.session_state.user_email = email
                        st.session_state.user_name = response.data[0].get('name', email)
                        st.session_state.user_role = response.data[0].get('role', 'user')
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
                except Exception as e:
                    st.error(f"Login error: {e}")
            else:
                st.warning("Please enter both email and password")

# ========== AGENT ACTION TYPES ==========
class ActionType(Enum):
    ALERT = "alert"
    REORDER = "reorder"
    PRICE_ADJUST = "price_adjust"
    FORECAST_UPDATE = "forecast_update"
    ANOMALY_DETECT = "anomaly_detect"


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AgentAction:
    action_type: ActionType
    product_id: int
    product_name: str
    description: str
    data: Dict
    timestamp: datetime
    alert_level: AlertLevel = AlertLevel.INFO
    executed: bool = False


# ========== SUPABASE & AI MODEL INITIALIZATION ==========
@st.cache_resource
def get_supabase_client():
    try:
        return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])
    except Exception as e:
        logger.error(f"Supabase connection failed: {e}")
        return None


# ========== VGG16 EMBEDDER ==========
@st.cache_resource
def get_resnet50_embedder():
    return ResNet50Embedder()

class ResNet50Embedder:
    def __init__(self):
        logger.info("Initializing ResNet50 embedder...")
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove final FC layer
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        logger.info(f"ResNet50 embedder ready on {self.device}")

    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.array([]).reshape(0, 2048).astype(np.float32)
        try:
            images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]
            tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
            with torch.no_grad():
                features = self.model(tensors).cpu().numpy().squeeze()
            if features.ndim == 1:
                features = features.reshape(1, -1)
            # Normalize for cosine similarity
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            features = features / norms
            return features.astype(np.float32)
        except Exception as e:
            logger.error(f"Error in ResNet50 embed_batch: {e}")
            return np.zeros((len(images), 2048), dtype=np.float32)

# ========== FAISS INDEXER ==========
class FaissIndexer:
    def __init__(self, dim: int = EMBED_DIM, index_path: Path = FAISS_INDEX_PATH):
        self.dim = dim
        self.index_path = index_path
        try:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
            else:
                self.index = self._create_new_index()
        except Exception:
            logger.exception("Failed to load FAISS index, creating a new one")
            self.index = self._create_new_index()

    def _create_new_index(self):
        base_index = faiss.IndexFlatL2(self.dim)
        return faiss.IndexIDMap(base_index)

    def save(self):
        try:
            faiss.write_index(self.index, str(self.index_path))
        except Exception:
            logger.exception("Failed to save FAISS index")

    def add(self, vectors: np.ndarray, ids: np.ndarray):
        try:
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            ids = np.asarray(ids, dtype=np.int64)
            if ids.ndim == 0:
                ids = np.array([ids], dtype=np.int64)
            self.index.add_with_ids(vectors, ids)
            self.save()
            logger.info(f"Added {len(vectors)} vectors to FAISS index")
        except Exception:
            logger.exception("Error adding vectors to FAISS index")

    def search(self, vectors: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if self.index.ntotal == 0:
                return np.array([]), np.array([])
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            top_k = min(top_k, max(1, int(self.index.ntotal)))
            distances, indices = self.index.search(vectors, top_k)
            return distances, indices
        except Exception:
            logger.exception("Error searching FAISS index")
            return np.array([]), np.array([])

    def reset(self):
        logger.info("Resetting FAISS index.")
        self.index = self._create_new_index()
        self.save()


@st.cache_resource
def get_faiss_indexer() -> FaissIndexer:
    return FaissIndexer()


# ========== IMAGE PROCESSOR ==========
@st.cache_resource
def get_image_processor():
    return ImageProcessor()


# ========== ENHANCED IMAGE PROCESSOR ==========
class ImageProcessor:
    """
    Enhanced with better object detection filtering
    """
    def __init__(self, detector_model: str = DEFAULT_DETECTOR_MODEL):
        try:
            self.detector = YOLO(detector_model)
            logger.info(f"YOLO detector loaded: {detector_model}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.detector = None

    def detect_and_crop(self, image: Image.Image, confidence: float = DEFAULT_CONFIDENCE) -> List[Image.Image]:
        """
        Enhanced detection with class filtering for products
        """
        if self.detector is None:
            return [image]
        
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            results = self.detector.predict(source=image, conf=confidence, verbose=False)
            crops = []
            
            # Define COCO classes that are likely to be products
            PRODUCT_CLASSES = {
                'bottle', 'cup', 'bowl', 'chair', 'couch', 'bed', 'table',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
                'snowboard', 'sports ball', 'kite', 'baseball bat', 'skateboard',
                'surfboard', 'tennis racket', 'wine glass', 'fork', 'knife', 'spoon'
            }
            
            detected_product_like_object = False
            
            if results and hasattr(results[0], 'boxes') and results[0].boxes and len(results[0].boxes) > 0:
                for idx, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
                    if hasattr(results[0].boxes, 'cls'):
                        class_id = int(results[0].boxes.cls[idx].item())
                        class_name = self.detector.names.get(class_id, '').lower()
                        
                        if class_name in ['person', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
                            logger.info(f"Skipping non-product class: {class_name}")
                            continue
                        
                        if class_name in PRODUCT_CLASSES or class_name == '':
                            detected_product_like_object = True
                    
                    x1, y1, x2, y2 = map(int, box)
                    padding_x = int((x2 - x1) * 0.1)
                    padding_y = int((y2 - y1) * 0.1)
                    x1, y1, x2, y2 = max(0, x1 - padding_x), max(0, y1 - padding_y), min(image.width, x2 + padding_x), min(image.height, y2 + padding_y)
                    
                    if x2 > x1 and y2 > y1:
                        crops.append(image.crop((x1, y1, x2, y2)))
            
            if detected_product_like_object or len(crops) == 0:
                crops.append(image)
            else:
                logger.info("Only non-product objects detected, returning empty")
                return []
            
            if len(crops) > 2:
                center_x, center_y = image.width // 2, image.height // 2
                quarter_w, quarter_h = image.width // 4, image.height // 4
                crops.append(image.crop((max(0, center_x - quarter_w), max(0, center_y - quarter_h), min(image.width, center_x + quarter_w), min(image.height, center_y + quarter_h))))
            
            logger.info(f"Generated {len(crops)} crops for analysis")
            return crops
            
        except Exception as e:
            logger.error(f"Error in detect_and_crop: {e}")
            return [image]

# ========== FORECASTING & MONITORING AGENTS ==========
class ForecastingAgent:
    def __init__(self):
        self.name = "Forecasting Agent"
    
    def generate_forecast(self, product_id: int, horizon: int = 3) -> Dict:
        try:
            sales_df = self._fetch_sales_data(product_id)
            if len(sales_df) < 12:
                return {'error': f"Only found {len(sales_df)} months of sales data. At least 12 are required.", 'historical': sales_df}
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(sales_df[['quantity']])
            look_back = 6
            X, y = self._create_sequences(scaled_data, look_back)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            model = Sequential([
                LSTM(50, return_sequences=True, activation='tanh', input_shape=(look_back, 1)),
                Dropout(0.2),
                LSTM(30, activation='tanh'),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=50, batch_size=4, verbose=0)
            
            future_predictions = []
            current_sequence = scaled_data[-look_back:].reshape(1, look_back, 1)
            
            for _ in range(horizon):
                pred_scaled = model.predict(current_sequence, verbose=0)[0][0]
                future_predictions.append(pred_scaled)
                current_sequence = np.append(current_sequence[:, 1:, :], [[[pred_scaled]]], axis=1)
            
            future_pred_denorm = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            
            return {
                'predictions': np.maximum(0, future_pred_denorm.flatten()),
                'confidence': 0.85,
                'trend': self._calculate_trend(sales_df['quantity'].values),
                'historical': sales_df.tail(24)
            }
        except Exception as e:
            logger.error(f"Forecast failed: {e}")
            return {'error': str(e), 'historical': pd.DataFrame()}
    
    def _fetch_sales_data(self, product_id: int) -> pd.DataFrame:
        response = supabase.table("sales_log").select("created_at, quantity_sold").eq("product_id", product_id).order("created_at").execute()
        if not response.data:
            return pd.DataFrame(columns=['date', 'quantity'])
        
        df = pd.DataFrame(response.data)
        df['date'] = pd.to_datetime(df['created_at'], format='ISO8601', utc=True).dt.tz_localize(None)
        df['quantity'] = pd.to_numeric(df['quantity_sold'])
        df.set_index('date', inplace=True)
        
        monthly_sales = df['quantity'].resample('M').sum().reset_index()
        
        if not monthly_sales.empty:
            full_date_range = pd.date_range(start=monthly_sales['date'].min(), end=datetime.now().replace(tzinfo=None), freq='M')
            monthly_sales.set_index('date', inplace=True)
            monthly_sales = monthly_sales.reindex(full_date_range, fill_value=0).reset_index().rename(columns={'index': 'date'})
        
        return monthly_sales
    
    def _create_sequences(self, data, look_back=6):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back), 0])
            y.append(data[i + look_back, 0])
        return np.array(X), np.array(y)
    
    def _calculate_trend(self, data: np.ndarray) -> str:
        if len(data) < 2: return "stable"
        recent, older = data[-3:].mean(), (data[-6:-3].mean() if len(data) >= 6 else data[:-3].mean())
        if recent > older * 1.1: return "increasing"
        if recent < older * 0.9: return "decreasing"
        return "stable"

class InventoryMonitorAgent:
    def __init__(self, forecasting_agent: ForecastingAgent):
        self.name, self.forecasting_agent = "Monitor Agent", forecasting_agent
    
    def monitor_all_products(self) -> List[AgentAction]:
        try:
            return [action for p in supabase.table("products").select("*").execute().data for action in self._analyze_product(p)]
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            return []
    
    def _analyze_product(self, product: Dict) -> List[AgentAction]:
        actions, stock, pid, name = [], float(product.get('stock_quantity', 0)), product['id'], product['name']
        if stock <= CRITICAL_STOCK_THRESHOLD:
            actions.append(AgentAction(ActionType.ALERT, pid, name, f"CRITICAL: Stock at {stock:.1f}", {'current_stock': stock}, datetime.now(), AlertLevel.CRITICAL))
            forecast = self.forecasting_agent.generate_forecast(pid, horizon=1)
            if 'predictions' in forecast:
                rec_order = max(20, forecast['predictions'][0] * REORDER_MULTIPLIER)
                actions.append(AgentAction(ActionType.REORDER, pid, name, f"Reorder: {rec_order:.0f} units", {'recommended_quantity': rec_order}, datetime.now(), AlertLevel.CRITICAL))
        elif stock <= LOW_STOCK_THRESHOLD:
            actions.append(AgentAction(ActionType.ALERT, pid, name, f"WARNING: Low stock at {stock:.1f}", {'current_stock': stock}, datetime.now(), AlertLevel.WARNING))
        return actions

class DecisionEngine:
    def __init__(self):
        self.name, self.forecasting_agent = "Decision Engine", ForecastingAgent()
        self.monitor_agent = InventoryMonitorAgent(self.forecasting_agent)
    
    def run_autonomous_cycle(self) -> List[AgentAction]:
        actions = self.monitor_agent.monitor_all_products()
        if actions: self._log_actions_to_db(actions)
        return actions
    
    def _log_actions_to_db(self, actions: List[AgentAction]):
        try:
            records = [{'action_type': a.action_type.value, 'product_id': a.product_id, 'description': a.description, 'data': json.dumps(a.data), 'alert_level': a.alert_level.value, 'created_at': a.timestamp.isoformat()} for a in actions]
            supabase.table("agent_actions").insert(records).execute()
        except Exception as e:
            logger.error(f"Failed to log actions: {e}")
    
    def get_recent_actions(self, limit: int = 50) -> List[Dict]:
        try:
            return supabase.rpc('get_recent_agent_actions', {'action_limit': limit}).execute().data or []
        except: return []
    
    def execute_reorder_action(self, action_id: int, reorder_qty: float) -> bool:
        try:
            action = supabase.table("agent_actions").select("product_id").eq("id", action_id).single().execute().data
            if not action: return False
            stock = supabase.table("products").select("stock_quantity").eq("id", action['product_id']).single().execute().data
            if stock:
                new_stock = float(stock['stock_quantity']) + reorder_qty
                supabase.table("products").update({'stock_quantity': new_stock}).eq("id", action['product_id']).execute()
                product = supabase.table("products").select("*").eq("id", action['product_id']).single().execute().data
                if product:
                    supabase.table("purchase_log").insert({'product_id': action['product_id'], 'quantity_purchased': reorder_qty, 'price_per_unit': product['price'], 'supplier': 'Auto Reorder'}).execute()
                supabase.rpc('mark_action_executed', {'action_id': action_id}).execute()
                return True
        except Exception as e:
            logger.error(f"Reorder failed: {e}")
        return False

# ========== AI ANALYTICS RESPONSE FUNCTION ==========
def get_ai_analytics_response(query: str) -> str:
    try:
        query_lower = query.lower()
        products = supabase.table("products").select("*").execute().data
        sales_data = supabase.table("sales_log").select("*, products(name, sku)").execute().data
        purchase_data = supabase.table("purchase_log").select("*, products(name, sku)").execute().data
        
        if any(word in query_lower for word in ['top', 'best', 'most selling', 'popular']):
            if not sales_data: return "I don't have enough sales data to answer that question yet."
            sales_df = pd.DataFrame(sales_data)
            sales_df['product_name'] = sales_df['products'].apply(lambda x: x['name'] if x else 'Unknown')
            top_products = sales_df.groupby('product_name')['quantity_sold'].sum().nlargest(5)
            response = "**Top 5 Selling Products:**\n\n" + "\n".join([f"{i}. {p}: {q:.2f} units sold" for i, (p, q) in enumerate(top_products.items(), 1)])
            return response
        
        if any(word in query_lower for word in ['profit', 'margin', 'earn']):
            if not sales_data or not purchase_data: return "I need both sales and purchase data to calculate profit."
            total_revenue = sum(float(s['quantity_sold']) * float(s['price_at_sale']) for s in sales_data)
            total_cost = sum(float(p['quantity_purchased']) * float(p.get('basic_price', p['price_per_unit'])) for p in purchase_data)
            profit = total_revenue - total_cost
            margin = (profit / total_revenue * 100) if total_revenue > 0 else 0
            return f"""**Profit Analysis:**\n\n- Total Revenue: {CURRENCY_SYMBOL}{total_revenue:,.2f}\n- Total Cost: {CURRENCY_SYMBOL}{total_cost:,.2f}\n- Gross Profit: {CURRENCY_SYMBOL}{profit:,.2f}\n- Profit Margin: {margin:.2f}%\n\n{'Excellent!' if margin > 30 else 'Good margin.' if margin > 20 else 'Consider improving margins.'}"""
        
        if any(word in query_lower for word in ['low stock', 'running low', 'need reorder', 'out of stock']):
            if not products: return "No products in inventory yet."
            low_stock = [p for p in products if float(p['stock_quantity']) <= LOW_STOCK_THRESHOLD]
            if not low_stock: return "Great news! All products have healthy stock levels."
            response = "**Products with Low Stock:**\n\n" + "\n".join([f"{'🔴 CRITICAL' if float(p['stock_quantity']) <= CRITICAL_STOCK_THRESHOLD else '🟡 LOW'} - {p['name']}: {p['stock_quantity']} units" for p in low_stock])
            return response

        if any(word in query_lower for word in ['average order', 'avg order', 'average sale']):
            if not sales_data: return "No sales data available yet."
            sales_df = pd.DataFrame(sales_data)
            sales_df['revenue'] = sales_df['quantity_sold'] * sales_df['price_at_sale']
            return f"**Average Order Value:** {CURRENCY_SYMBOL}{sales_df['revenue'].mean():.2f}\n\nThis is the average revenue per transaction."
        
        if any(word in query_lower for word in ['how many sales', 'sales count', 'number of sales']):
            if not sales_data: return "No sales recorded yet."
            sales_df = pd.DataFrame(sales_data); sales_df['date'] = pd.to_datetime(sales_df['created_at'])
            if '7 day' in query_lower or 'week' in query_lower:
                recent = sales_df[sales_df['date'] >= (datetime.now() - pd.Timedelta(days=7))]
                return f"**Sales in Last 7 Days:** {len(recent)} transactions\n\nTotal revenue: {CURRENCY_SYMBOL}{(recent['quantity_sold'] * recent['price_at_sale']).sum():.2f}"
            elif 'month' in query_lower:
                recent = sales_df[sales_df['date'] >= (datetime.now() - pd.Timedelta(days=30))]
                return f"**Sales in Last 30 Days:** {len(recent)} transactions\n\nTotal revenue: {CURRENCY_SYMBOL}{(recent['quantity_sold'] * recent['price_at_sale']).sum():.2f}"
            else:
                return f"**Total Sales:** {len(sales_df)} transactions\n\nTotal revenue: {CURRENCY_SYMBOL}{(sales_df['quantity_sold'] * sales_df['price_at_sale']).sum():.2f}"
        
        return "I can help you analyze sales, profit margins, inventory levels, and trends. Try asking about top products, profit analysis, or low stock items!"
    except Exception as e:
        logger.exception("Failed to generate AI analytics response")
        return f"I encountered an error analyzing the data: {str(e)}"

# ========== INVENTORY MANAGEMENT SYSTEM ==========
class InventoryManagementSystem:
    def __init__(self):
        if not supabase: raise RuntimeError("Supabase client is not available.")
        self.embedder, self.indexer, self.processor, self.decision_engine = get_resnet50_embedder(), get_faiss_indexer(), get_image_processor(), DecisionEngine()
    
    def add_product(self, name, sku, price, stock, image_bytes, category="", description=""):
        try:
            res = supabase.table("products").insert({"name": name, "sku": sku, "price": price, "stock_quantity": stock, "category": category, "description": description}).execute()
            if not res.data: st.error("Failed to insert product."); return None
            product = res.data[0]
            self.link_product_image(product["id"], sku, image_bytes, is_new_product=True)
            return product
        except Exception as e:
            st.error(f"Error adding product: {e}"); return None

    def link_product_image(self, pid, sku, image_bytes, is_new_product=False):
        try:
            file_name = f"{pid}/{sku}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png" if not is_new_product else f"{pid}/{sku}.png"
            supabase.storage.from_(BUCKET_NAME).upload(file_name, image_bytes, {"content-type": "image/png", "upsert": "true"})
            if is_new_product:
                supabase.table("products").update({"image_url": supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)}).eq("id", pid).execute()
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            embs = self.embedder.embed_batch(self.processor.detect_and_crop(img))
            if embs.shape[0] > 0:
                self.indexer.add(embs, np.full((embs.shape[0],), pid, dtype=np.int64))
            return True
        except Exception as e:
            logger.error(f"Error linking image: {e}"); st.error(f"Error linking image: {e}"); return False
    
    def search_product_by_image(self, image_bytes, top_k=20):
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            crops = self.processor.detect_and_crop(img)
            if not crops: return None
            embs = self.embedder.embed_batch(crops)
            if embs.shape[0] == 0: return None
            distances, ids = self.indexer.search(embs, top_k)
            
            product_votes = {}
            EXCELLENT_MATCH, GOOD_MATCH, ACCEPTABLE_MATCH = 15.0, 25.0, 35.0
            for i in range(len(embs)):
                for dist, pid in zip(distances[i], ids[i]):
                    if pid != -1 and dist < ACCEPTABLE_MATCH:
                        if pid not in product_votes:
                            product_votes[pid] = {'count': 0, 'total_distance': 0, 'min_distance': dist, 'excellent_matches': 0, 'good_matches': 0}
                        stats = product_votes[pid]
                        stats['count'] += 1; stats['total_distance'] += dist; stats['min_distance'] = min(stats['min_distance'], dist)
                        if dist < EXCELLENT_MATCH: stats['excellent_matches'] += 1
                        elif dist < GOOD_MATCH: stats['good_matches'] += 1
            
            if not product_votes: logger.info("No products matched"); return None
            
            scored_products = sorted([(pid, (s['excellent_matches'] * 3 + s['good_matches'] * 2 + s['count']) / (s['total_distance'] / s['count'] + 0.1), s['count'], s['min_distance'], s['excellent_matches'], s['good_matches']) for pid, s in product_votes.items()], key=lambda x: x[1], reverse=True)
            best_id, best_score, vote_count, min_dist, excellent, good = scored_products[0]
            
            logger.info(f"Best match: ID {best_id}, Score: {best_score:.4f}, Votes: {vote_count}, Min Dist: {min_dist:.4f}")
            if (excellent < 1 and good < 2) or min_dist > GOOD_MATCH or (min_dist > EXCELLENT_MATCH and vote_count < 2) or (len(scored_products) > 1 and scored_products[1][1] > best_score * 0.8):
                logger.info("Match rejected by validation rules."); return None

            res = supabase.table("products").select("*").eq("id", best_id).execute()
            if res.data:
                p = res.data[0]
                p.update({'match_count': vote_count, 'match_score': best_score, 'match_confidence': 'EXCELLENT' if excellent >= 2 else 'GOOD' if good >= 2 else 'FAIR', 'min_distance': min_dist})
                return p
        except Exception as e:
            logger.error(f"Image search failed: {e}"); st.error(f"Image search failed: {e}")
        return None

    def record_sale(self, product, quantity, price):
        try:
            if float(product['stock_quantity']) < quantity: st.warning("Not enough stock."); return False
            supabase.table("products").update({"stock_quantity": float(product['stock_quantity']) - quantity}).eq("id", product['id']).execute()
            supabase.table("sales_log").insert({"product_id": product['id'], "quantity_sold": quantity, "price_at_sale": price}).execute()
            return True
        except Exception as e:
            st.error(f"Sale recording failed: {e}"); return False
    
    def record_purchase_detailed(self, product_id, quantity, basic_price, supplier, invoice_no, hsn_code, sgst_percent, cgst_percent, igst_percent, round_off, total):
        try:
            stock = supabase.table("products").select("stock_quantity").eq("id", product_id).single().execute().data
            if stock:
                supabase.table("products").update({'stock_quantity': float(stock['stock_quantity']) + quantity}).eq("id", product_id).execute()
            supabase.table("purchase_log").insert({'product_id': product_id, 'quantity_purchased': quantity, 'price_per_unit': basic_price, 'basic_price': basic_price, 'sgst_percent': sgst_percent, 'cgst_percent': cgst_percent, 'igst_percent': igst_percent, 'round_off': round_off, 'total_amount': total, 'supplier': supplier, 'invoice_no': invoice_no, 'hsn_code': hsn_code}).execute()
            return True
        except Exception as e:
            st.error(f"Purchase recording failed: {e}"); return False
    
    def delete_product_image(self, file_path: str):
        try:
            supabase.storage.from_(BUCKET_NAME).remove([file_path]); return True
        except Exception as e:
            st.error(f"Failed to delete image {file_path}: {e}"); return False
    
    def rebuild_faiss_index(self):
        try:
            logger.info("Starting FAISS index rebuild..."); self.indexer.reset()
            all_products = supabase.table("products").select("id, sku").execute().data
            if not all_products: return True
            all_embeddings, all_ids = [], []
            for p in all_products:
                try:
                    for file_info in supabase.storage.from_(BUCKET_NAME).list(str(p['id'])):
                        try:
                            image_bytes = supabase.storage.from_(BUCKET_NAME).download(f"{p['id']}/{file_info['name']}")
                            if image_bytes:
                                embs = self.embedder.embed_batch(self.processor.detect_and_crop(Image.open(io.BytesIO(image_bytes)).convert("RGB")))
                                all_embeddings.extend(embs); all_ids.extend([p['id']] * len(embs))
                        except Exception as img_e: logger.error(f"Failed to process image: {img_e}")
                except Exception as prod_e: logger.error(f"Failed to process product {p['id']}: {prod_e}")
            if all_embeddings:
                self.indexer.add(np.array(all_embeddings, dtype=np.float32), np.array(all_ids, dtype=np.int64))
                logger.info(f"Rebuilt index with {len(all_embeddings)} embeddings from {len(all_products)} products")
            return True
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}"); st.error(f"Failed to rebuild FAISS index: {e}"); return False
    
    def delete_product(self, product_id: int):
        try:
            files = supabase.storage.from_(BUCKET_NAME).list(str(product_id))
            if files: supabase.storage.from_(BUCKET_NAME).remove([f"{product_id}/{f['name']}" for f in files])
            supabase.table("products").delete().eq("id", product_id).execute()
            return self.rebuild_faiss_index()
        except Exception as e:
            st.error(f"Failed to delete product: {e}"); return False

# ========== DATA LOADING FUNCTIONS ==========
@st.cache_data
def load_and_prepare_csv(file):
    df = pd.read_csv(file)
    df.rename(columns={'DATE': 'd', 'PRODUCT ID': 's', 'NAME': 'n', 'QUANTITY': 'q', 'PRICE': 'p'}, inplace=True)
    df.update(pd.DataFrame({'d': pd.to_datetime(df['d'], format='%m/%d/%Y'), 's': df['s'].astype(str), 'q': pd.to_numeric(df['q'], errors='coerce').fillna(0.0), 'p': pd.to_numeric(df['p'], errors='coerce').fillna(0.0)}))
    return df

def sync_historical_data(df):
    st.info("Syncing historical data..."); progress = st.progress(0)
    existing = {p['sku'] for p in supabase.table('products').select('sku').execute().data}
    new_products = [{'sku': r['s'], 'name': r['n'], 'price': r['p'], 'stock_quantity': 0} for _, r in df.sort_values('d').drop_duplicates(subset=['s'], keep='last').iterrows() if r['s'] not in existing]
    if new_products: supabase.table('products').insert(new_products).execute()
    progress.progress(0.25, "Products synced.")
    sku_map = {p['sku']: p['id'] for p in supabase.table('products').select('id, sku').execute().data}
    sales = []
    for i, r in df.iterrows():
        if r['s'] in sku_map: sales.append({'product_id': sku_map[r['s']], 'quantity_sold': r['q'], 'price_at_sale': r['p'], 'created_at': r['d'].isoformat()})
        if (i + 1) % 500 == 0 or (i + 1) == len(df):
            if sales: supabase.table('sales_log').insert(sales).execute(); sales = []
            progress.progress(0.25 + (0.75 * (i + 1) / len(df)), f"Synced {i+1}/{len(df)} sales.")
    st.success("Sync complete!")

# ========== PAGE FUNCTIONS ==========
def show_agent_dashboard(system):
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Autonomous Agent Dashboard</h1>
        <p>Real-time monitoring of AI agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    actions = system.decision_engine.get_recent_actions(limit=50)
    
    if not actions:
        with st.spinner("Running initial agent analysis..."):
            system.decision_engine.run_autonomous_cycle()
            actions = system.decision_engine.get_recent_actions(limit=50)
    
    if actions:
        st.markdown("### 📊 Quick Stats")
        c1, c2, c3, c4 = st.columns(4)
        total_actions = len(actions)
        critical = len([a for a in actions if a.get('alert_level') == 'critical'])
        warnings = len([a for a in actions if a.get('alert_level') == 'warning'])
        reorders = len([a for a in actions if a.get('action_type') == 'reorder'])
        
        c1.metric("Total Actions", total_actions)
        c2.metric("Critical", critical, delta=f"-{critical}" if critical > 0 else "0", delta_color="inverse")
        c3.metric("Warnings", warnings, delta=f"-{warnings}" if warnings > 0 else "0", delta_color="inverse")
        c4.metric("Reorder Needed", reorders)
        
        st.markdown("---")
        st.markdown("### 📋 Action Items")
        
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            filter_level = st.selectbox("Filter by Alert Level", ["All", "critical", "warning", "info"])
        with filter_col2:
            filter_type = st.selectbox("Filter by Action Type", ["All", "alert", "reorder", "price_adjust", "forecast_update"])
        
        filtered_actions = actions
        if filter_level != "All":
            filtered_actions = [a for a in filtered_actions if a.get('alert_level') == filter_level]
        if filter_type != "All":
            filtered_actions = [a for a in filtered_actions if a.get('action_type') == filter_type]
        
        for a in filtered_actions:
            level = a.get('alert_level', 'info')
            icon = "🔴" if level == "critical" else "🟡" if level == "warning" else "🟢"
            
            with st.expander(f"{icon} {a.get('action_type', 'N/A').upper()} - {a.get('product_name', 'Unknown')} - {a.get('created_at', '')[:19]}"):
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.write(f"**Description:** {a.get('description', 'N/A')}")
                    st.json(a.get('data', '{}'))
                
                with c2:
                    st.metric("Alert Level", level.upper())
                    st.metric("Executed", "Yes ✓" if a.get('executed') else "No ✗")
                    
                    if a.get('action_type') == 'reorder' and not a.get('executed'):
                        data = json.loads(a.get('data', '{}')) if isinstance(a.get('data'), str) else a.get('data', {})
                        qty = st.number_input(
                            "Reorder Qty",
                            0.0,
                            value=float(data.get('recommended_quantity', 20)),
                            key=f"qty_{a['id']}"
                        )
                        if st.button("Execute Reorder", key=f"exec_{a['id']}", type="primary"):
                            if system.decision_engine.execute_reorder_action(a['id'], qty):
                                st.success("Reorder executed!")
                                st.rerun()
    else:
        st.info("No agent actions yet. The system will generate actions as it monitors your inventory.")


def show_user_management():
    st.markdown("""
    <div class="main-header">
        <h1>👥 User Management</h1>
        <p>Manage system users and permissions</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.get('user_role') != 'admin':
        st.error("🚫 Access Denied: Only administrators can access this page.")
        return
    
    tab1, tab2, tab3 = st.tabs(["📋 All Users", "➕ Add User", "🗑️ Delete User"])
    
    with tab1:
        st.subheader("Registered Users")
        try:
            users = supabase.table("users").select("*").order("created_at", desc=True).execute().data
            
            if users:
                users_display = []
                for user in users:
                    users_display.append({
                        'ID': user.get('id'),
                        'Name': user.get('name'),
                        'Email': user.get('email'),
                        'Role': user.get('role', 'user').upper(),
                        'Created': pd.to_datetime(user.get('created_at')).strftime('%Y-%m-%d %H:%M') if user.get('created_at') else 'N/A'
                    })
                
                users_df = pd.DataFrame(users_display)
                st.dataframe(users_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Users", len(users))
                col2.metric("Admins", len([u for u in users if u.get('role') == 'admin']))
                col3.metric("Regular Users", len([u for u in users if u.get('role') != 'admin']))
            else:
                st.info("No users found in the system.")
        except Exception as e:
            st.error(f"Error loading users: {e}")
    
    with tab2:
        st.subheader("Add New User")
        
        with st.form("admin_add_user_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("Full Name *", placeholder="John Doe")
                new_email = st.text_input("Email Address *", placeholder="john@example.com")
            with col2:
                new_password = st.text_input("Password *", type="password", placeholder="Minimum 6 characters")
                new_role = st.selectbox("User Role *", ["user", "admin"])
            
            submit_new_user = st.form_submit_button("✅ Create User", type="primary", use_container_width=True)
        
        if submit_new_user:
            if all([new_name, new_email, new_password]):
                if len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters long!")
                else:
                    try:
                        existing = supabase.table("users").select("email").eq("email", new_email).execute()
                        
                        if existing.data and len(existing.data) > 0:
                            st.error("❌ Email already exists in the system!")
                        else:
                            hashed_pw = hash_password(new_password)
                            
                            new_user = {
                                "name": new_name,
                                "email": new_email,
                                "password_hash": hashed_pw,
                                "role": new_role
                            }
                            
                            response = supabase.table("users").insert(new_user).execute()
                            
                            if response.data:
                                st.success(f"✅ User '{new_name}' created successfully!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("❌ Failed to create user. Please try again.")
                    except Exception as e:
                        st.error(f"Error creating user: {e}")
            else:
                st.warning("⚠️ Please fill all required fields (marked with *)")
    
    with tab3:
        st.subheader("Delete User")
        st.warning("⚠️ Deleting a user is permanent and cannot be undone!")
        
        try:
            users = supabase.table("users").select("id, name, email").execute().data
            
            if users and len(users) > 0:
                current_email = st.session_state.get('user_email')
                deletable_users = [u for u in users if u.get('email') != current_email]
                
                if deletable_users:
                    user_options = {u['id']: f"{u['name']} ({u['email']})" for u in deletable_users}
                    selected_user_id = st.selectbox(
                        "Select User to Delete",
                        options=list(user_options.keys()),
                        format_func=lambda x: user_options[x]
                    )
                    
                    confirm_text = f"DELETE {user_options[selected_user_id]}"
                    user_input = st.text_input(f"Type '{confirm_text}' to confirm:", key="delete_user_confirm")
                    
                    if user_input == confirm_text:
                        if st.button("🗑️ Delete User Permanently", type="primary"):
                            try:
                                supabase.table("users").delete().eq("id", selected_user_id).execute()
                                st.success("✅ User deleted successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting user: {e}")
                    else:
                        st.info("Please type the confirmation text exactly to enable deletion.")
                else:
                    st.info("No other users available to delete. You cannot delete your own account.")
            else:
                st.info("No users found in the system.")
        except Exception as e:
            st.error(f"Error loading users: {e}")


def show_point_of_sale(system):
    st.markdown("""
    <div class="main-header">
        <h1>🛍️ Smart Point of Sale</h1>
        <p>Upload a product image to sell or add it</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add helpful tips
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        **For accurate product detection:**
        - 📸 Take clear, well-lit photos of the product
        - 🎯 Focus on the product itself, minimize background clutter
        - 📏 Ensure the product is the main subject in the image
        - ⚠️ Avoid images with people, animals, or multiple different products
        - 🔄 If first attempt fails, try a different angle or closer shot
        """)
    
    img = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"], key="pos_uploader")
    
    if img:
        bytes_data = img.getvalue()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(bytes_data, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            with st.spinner("🔍 Analyzing image and identifying products..."):
                product = system.search_product_by_image(bytes_data)
            
            if product:
                # Show confidence level with color coding
                confidence = product.get('match_confidence', 'FAIR')
                confidence_colors = {
                    'EXCELLENT': '🟢',
                    'GOOD': '🟡', 
                    'FAIR': '🟠'
                }
                
                st.success(f"{confidence_colors.get(confidence, '🟢')} **Match Found:** {product['name']}")
                
                # Show detailed match info
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Confidence", confidence)
                col_b.metric("Match Quality", f"{product.get('match_count', 1)} detections")
                col_c.metric("Distance Score", f"{product.get('min_distance', 0):.1f}")
                
                if confidence == 'FAIR':
                    st.warning("⚠️ Match confidence is FAIR. Please verify this is the correct product.")
                
                # Product details
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("SKU", product['sku'])
                c2.metric("Price", f"{CURRENCY_SYMBOL}{product['price']:.2f}")
                c3.metric("Stock", f"{float(product['stock_quantity']):.2f} units")
                
                if product.get('category'):
                    st.info(f"**Category:** {product['category']}")
                
                if product.get('description'):
                    st.info(f"**Description:** {product['description']}")
                
                # Show product image if available
                if product.get('image_url'):
                    with st.expander("📷 View Stored Product Image"):
                        st.image(product['image_url'], caption="Product in Inventory", width=300)
                
                # Stock alerts
                if float(product['stock_quantity']) <= CRITICAL_STOCK_THRESHOLD:
                    st.error(f"🔴 CRITICAL: Only {product['stock_quantity']} units remaining!")
                elif float(product['stock_quantity']) <= LOW_STOCK_THRESHOLD:
                    st.warning(f"🟡 LOW STOCK: {product['stock_quantity']} units remaining")
                
                # Sale form
                with st.form("sale_form"):
                    st.subheader("Record Sale")
                    qty = st.number_input("Quantity to Sell", 0.01, float(product['stock_quantity']), 1.0, 0.1, format="%.2f")
                    price = st.number_input(f"Price per Unit ({CURRENCY_SYMBOL})", 0.01, value=float(product['price']), format="%.2f")
                    
                    total_amount = qty * price
                    st.info(f"**Total Amount:** {CURRENCY_SYMBOL}{total_amount:.2f}")
                    
                    submitted = st.form_submit_button("🛒 Record Sale", type="primary", use_container_width=True)
                
                if submitted:
                    if system.record_sale(product, qty, price):
                        st.balloons()
                        st.success("✅ Sale recorded successfully!")
                        system.decision_engine.run_autonomous_cycle()
                        st.rerun()
            else:
                # No match found - provide clear feedback
                st.error("❌ No matching product found in inventory")
                
                st.info("""
                **Possible reasons:**
                - This is a new product not yet in inventory
                - Image quality is too low or unclear
                - Product is not clearly visible in the image
                - Image contains non-product items (people, animals, scenery)
                - Background is too cluttered
                
                **Try:**
                1. 📸 Taking a clearer, closer photo of just the product
                2. ✨ Ensuring good lighting
                3. 🎯 Removing background clutter
                4. ➕ Adding this as a new product below
                """)
                
                st.markdown("---")
                
                # Add new product form
                with st.form("add_product_form"):
                    st.subheader("➕ Add New Product to Inventory")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        name = st.text_input("Product Name *", placeholder="e.g., Cement Bag 50kg")
                        sku = st.text_input("SKU / Product Code *", placeholder="e.g., CEM-50-001")
                        category = st.text_input("Category", placeholder="e.g., Construction Materials")
                    
                    with col2:
                        price = st.number_input(f"Selling Price ({CURRENCY_SYMBOL}) *", 0.01, format="%.2f")
                        stock = st.number_input("Initial Stock Quantity *", 0.0, step=1.0, format="%.2f")
                    
                    description = st.text_area("Product Description (Optional)", 
                                                placeholder="Describe the product features, specifications, etc.")
                    
                    st.info("💡 This uploaded image will be used to train the AI for future recognition")
                    
                    submit_btn = st.form_submit_button("✅ Add Product to Inventory", type="primary", use_container_width=True)
                
                if submit_btn:
                    if all([name, sku, price is not None, stock is not None]):
                        # Check for duplicate SKU
                        existing = supabase.table("products").select("sku").eq("sku", sku).execute()
                        if existing.data and len(existing.data) > 0:
                            st.error(f"❌ A product with SKU '{sku}' already exists. Please use a unique SKU.")
                        else:
                            with st.spinner("Adding product to inventory and training AI..."):
                                result = system.add_product(name, sku, price, stock, bytes_data, category, description)
                                if result:
                                    st.success(f"✅ Successfully added '{name}' to inventory!")
                                    st.info("🤖 AI has been trained to recognize this product in future uploads")
                                    st.balloons()
                                    st.rerun()
                    else:
                        st.error("❌ Please fill all required fields (marked with *)")
    
    # Show recent sales
    st.markdown("---")
    st.subheader("📊 Recent Sales Today")
    
    try:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        recent_sales = supabase.table("sales_log").select("*, products(name)").gte("created_at", today_start.isoformat()).order("created_at", desc=True).limit(5).execute()
        
        if recent_sales.data and len(recent_sales.data) > 0:
            for sale in recent_sales.data:
                product_name = sale['products']['name'] if sale.get('products') else 'Unknown'
                sale_time = pd.to_datetime(sale['created_at']).strftime('%H:%M')
                amount = float(sale['quantity_sold']) * float(sale['price_at_sale'])
                st.info(f"🕐 {sale_time} | {product_name} | Qty: {sale['quantity_sold']} | {CURRENCY_SYMBOL}{amount:.2f}")
        else:
            st.info("No sales recorded today yet")
    except Exception as e:
        logger.error(f"Error loading recent sales: {e}")
        
def show_inventory_management(system):
    st.markdown("""
    <div class="main-header">
        <h1>📦 Inventory Management</h1>
        <p>Manage your products and inventory</p>
    </div>
    """, unsafe_allow_html=True)
    
    t1, t2, t3, t4, t5 = st.tabs(["📋 View All", "✏️ Update Stock/Price", "🖼️ Link Images", "📊 Sales Log", "🗑️ Delete Product"])
    
    with t1:
        st.subheader("Current Inventory")
        data = supabase.table("products").select("*").order("name").execute()
        
        if data.data:
            df = pd.DataFrame(data.data)
            df['status'] = df['stock_quantity'].apply(lambda x: '🔴' if float(x) <= CRITICAL_STOCK_THRESHOLD else '🟡' if float(x) <= LOW_STOCK_THRESHOLD else '🟢')
            df['price'] = df['price'].apply(lambda x: f"{CURRENCY_SYMBOL}{float(x):.2f}")
            df['stock_quantity'] = df['stock_quantity'].apply(lambda x: f"{float(x):.2f}")
            
            display_df = df[['status', 'name', 'sku', 'category', 'stock_quantity', 'price', 'description']]
            display_df.columns = ['Status', 'Product Name', 'SKU', 'Category', 'Stock', 'Price', 'Description']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Summary metrics
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Products", len(df))
            col2.metric("Low Stock", len(df[df['status'] == '🟡']))
            col3.metric("Critical Stock", len(df[df['status'] == '🔴']))
            col4.metric("Healthy Stock", len(df[df['status'] == '🟢']))
        else:
            st.info("No products in inventory yet.")
    
    with t2:
        st.subheader("Update Stock and Price")
        products = supabase.table("products").select("id, name, sku, stock_quantity, price").order("name").execute().data
        
        if products:
            opts = {p['id']: f"{p['name']} ({p['sku']})" for p in products}
            selected_id = st.selectbox("Select a Product", options=list(opts.keys()), format_func=lambda x: opts[x], key="update_product_select")
            selected = next((p for p in products if p['id'] == selected_id), None)
            
            if selected:
                c1, c2 = st.columns(2)
                c1.metric("Current Stock", f"{float(selected['stock_quantity']):.2f}")
                c2.metric("Current Price", f"{CURRENCY_SYMBOL}{float(selected['price']):.2f}")
                
                with st.form("update_stock_price_form", clear_on_submit=True):
                    qty_add = st.number_input("Quantity to Add", 0.0, step=1.0, format="%.2f", key="qty_add_input")
                    new_price = st.number_input(f"New Price ({CURRENCY_SYMBOL}) (optional)", 0.01, value=float(selected['price']), format="%.2f", key="new_price_input")
                    submit_update = st.form_submit_button("✅ Apply Changes", type="primary", use_container_width=True)
                
                if submit_update:
                    payload = {}
                    if qty_add > 0:
                        payload['stock_quantity'] = float(selected['stock_quantity']) + qty_add
                    if new_price > 0 and abs(new_price - float(selected['price'])) > 0.01:
                        payload['price'] = new_price
                    if payload:
                        supabase.table('products').update(payload).eq('id', selected_id).execute()
                        st.success("✅ Product updated successfully!")
                        st.rerun()
                    else:
                        st.warning("⚠️ No changes to apply")
        else:
            st.info("No products available.")
    
    with t3:
        st.subheader("Link/Update Product Images")
        all_products = supabase.table("products").select("*").order("name").execute().data
        
        if all_products:
            for p in all_products:
                with st.expander(f"**{p['name']}** (SKU: `{p['sku']}`)"):
                    try:
                        files = supabase.storage.from_(BUCKET_NAME).list(str(p['id']))
                        
                        if files and len(files) > 0:
                            st.write(f"**Current Images:** {len(files)} image(s)")
                            
                            num_cols = min(5, len(files))
                            cols = st.columns(num_cols)
                            for i, file in enumerate(files):
                                with cols[i % num_cols]:
                                    try:
                                        url = supabase.storage.from_(BUCKET_NAME).get_public_url(f"{p['id']}/{file['name']}")
                                        st.image(url, width=100, caption=file['name'][:20])
                                        if st.button("🗑️", key=f"del_{p['id']}_{file['name']}", help="Delete this image"):
                                            if system.delete_product_image(f"{p['id']}/{file['name']}"):
                                                with st.spinner("Rebuilding search index..."):
                                                    system.rebuild_faiss_index()
                                                st.success("Image deleted!")
                                                st.rerun()
                                    except Exception as e:
                                        st.error(f"Error loading image: {e}")
                        else:
                            st.info("No images linked to this product yet.")
                        
                        img_files = st.file_uploader(
                            f"Upload more images for {p['name']}", 
                            ['png', 'jpg', 'jpeg'], 
                            key=f"up_{p['id']}", 
                            accept_multiple_files=True
                        )
                        
                        if img_files:
                            with st.spinner(f"Uploading {len(img_files)} image(s)..."):
                                for img_file in img_files:
                                    system.link_product_image(p['id'], p['sku'], img_file.getvalue())
                                with st.spinner("Rebuilding search index..."):
                                    system.rebuild_faiss_index()
                                st.success(f"✅ Added {len(img_files)} image(s) for {p['name']}!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error accessing storage: {e}")
        else:
            st.info("No products available.")
    
    with t4:
        st.subheader("📊 Historical Sales Log")
        log = supabase.table("sales_log").select("*, products(name, sku)").order("created_at", desc=True).limit(1000).execute().data
        
        if log:
            flat = []
            for r in log:
                if r.get('products'):
                    flat.append({
                        'Date': pd.to_datetime(r['created_at']).strftime('%Y-%m-%d %H:%M'),
                        'Product': r['products']['name'],
                        'SKU': r['products']['sku'],
                        'Quantity': f"{r['quantity_sold']:.2f}",
                        'Unit Price': f"{CURRENCY_SYMBOL}{r['price_at_sale']:.2f}",
                        'Total': f"{CURRENCY_SYMBOL}{float(r['quantity_sold']) * float(r['price_at_sale']):.2f}"
                    })
            
            if flat:
                sales_df = pd.DataFrame(flat)
                st.dataframe(sales_df, use_container_width=True, hide_index=True)
                
                csv = sales_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Sales Log",
                    data=csv,
                    file_name=f'sales_log_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv',
                    type="primary"
                )
            else:
                st.info("No sales with product information found.")
        else:
            st.info("No sales recorded yet.")
    
    with t5:
        st.subheader("⚠️ Permanently Delete a Product")
        st.warning("⚠️ This action is irreversible and will delete the product, its stock, and all associated images.")
        products = supabase.table("products").select("id, name, sku").order("name").execute().data
        
        if products:
            opts = {p['id']: f"{p['name']} ({p['sku']})" for p in products}
            selected_id = st.selectbox("Select Product to Delete", options=list(opts.keys()), format_func=lambda x: opts[x], key="delete_product_select")
            
            confirm_text = f"DELETE {opts[selected_id]}"
            user_input = st.text_input(f"Type '{confirm_text}' to confirm deletion:", key="delete_confirm")
            
            if user_input == confirm_text:
                if st.button("🗑️ Delete Product Permanently", type="primary"):
                    with st.spinner("Deleting product and rebuilding index..."):
                        if system.delete_product(selected_id):
                            st.success("✅ Product deleted successfully.")
                            st.rerun()
            else:
                st.info(f"Please type the confirmation text exactly to enable deletion.")
        else:
            st.info("No products available.")


def show_purchase_management(system):
    st.markdown("""
    <div class="main-header">
        <h1>📥 Purchase Management</h1>
        <p>Record and track purchases</p>
    </div>
    """, unsafe_allow_html=True)
    
    t1, t2 = st.tabs(["📝 Record Purchase", "📚 Purchase History"])
    
    with t1:
        st.subheader("Record New Purchase")
        products = supabase.table("products").select("id, name, sku, price").order("name").execute().data
        
        if products:
            opts = {p['id']: f"{p['name']} ({p['sku']})" for p in products}
            selected_id = st.selectbox("Select Product", options=list(opts.keys()), format_func=lambda x: opts[x], key="purchase_product_select")
            
            with st.form("purchase_entry_form", clear_on_submit=True):
                col1, col2 = st.columns(2)
                with col1:
                    invoice_no = st.text_input("Invoice Number *")
                    hsn_code = st.text_input("HSN Code *")
                with col2:
                    qty = st.number_input("Quantity Purchased *", 0.01, step=1.0, format="%.2f")
                    basic_price = st.number_input(f"Basic Price ({CURRENCY_SYMBOL}) *", 0.01, format="%.2f")
                
                col3, col4, col5 = st.columns(3)
                with col3:
                    sgst = st.number_input("SGST (%)", 0.0, 100.0, 9.0, format="%.2f")
                with col4:
                    cgst = st.number_input("CGST (%)", 0.0, 100.0, 9.0, format="%.2f")
                with col5:
                    igst = st.number_input("IGST (%)", 0.0, 100.0, 0.0, format="%.2f")
                
                col6, col7 = st.columns(2)
                with col6:
                    round_off = st.number_input("Round Off", -100.0, 100.0, 0.0, format="%.2f")
                with col7:
                    supplier = st.text_input("Supplier Name *")
                
                if basic_price > 0 and qty > 0:
                    sgst_amt = (basic_price * qty * sgst) / 100
                    cgst_amt = (basic_price * qty * cgst) / 100
                    igst_amt = (basic_price * qty * igst) / 100
                    calculated_total = (basic_price * qty) + sgst_amt + cgst_amt + igst_amt + round_off
                    st.info(f"**Calculated Total:** {CURRENCY_SYMBOL}{calculated_total:.2f}")
                
                submit_purchase = st.form_submit_button("💾 Record Purchase", type="primary", use_container_width=True)
            
            if submit_purchase:
                if qty > 0 and basic_price > 0 and supplier and invoice_no and hsn_code:
                    sgst_amt = (basic_price * qty * sgst) / 100
                    cgst_amt = (basic_price * qty * cgst) / 100
                    igst_amt = (basic_price * qty * igst) / 100
                    total = (basic_price * qty) + sgst_amt + cgst_amt + igst_amt + round_off
                    
                    if system.record_purchase_detailed(selected_id, qty, basic_price, supplier, 
                                                    invoice_no, hsn_code, sgst, cgst, igst, round_off, total):
                        st.success("✅ Purchase recorded successfully!")
                        st.rerun()
                else:
                    st.error("❌ Please fill all required fields (marked with *)")
        else:
            st.info("No products available. Please add products first.")
    
    with t2:
        st.subheader("📚 Purchase Book")
        purchases = supabase.table("purchase_log").select("*, products(name, sku)").order("created_at", desc=True).limit(1000).execute().data
        
        if purchases:
            flat = []
            for r in purchases:
                product_name = r['products']['name'] if r.get('products') else 'Unknown'
                flat.append({
                    'DATE': pd.to_datetime(r['created_at']).strftime('%d/%m/%Y'),
                    'I.NO': r.get('invoice_no', 'N/A'),
                    'HSN': r.get('hsn_code', 'N/A'),
                    'NAME': product_name,
                    'QTY': f"{r['quantity_purchased']:.2f}",
                    'BASIC': f"{CURRENCY_SYMBOL}{r.get('basic_price', r['price_per_unit']):.2f}",
                    'SGST': f"{r.get('sgst_percent', 0):.1f}%",
                    'CGST': f"{r.get('cgst_percent', 0):.1f}%",
                    'IGST': f"{r.get('igst_percent', 0):.1f}%",
                    'ROUND OFF': f"{CURRENCY_SYMBOL}{r.get('round_off', 0):.2f}",
                    'TOTAL': f"{CURRENCY_SYMBOL}{r.get('total_amount', float(r['quantity_purchased']) * float(r['price_per_unit'])):.2f}",
                    'SUPPLIER': r['supplier']
                })
            
            df_display = pd.DataFrame(flat)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("📊 Purchase Summary")
            total_purchases = len(purchases)
            total_spent = sum([r.get('total_amount', float(r['quantity_purchased']) * float(r['price_per_unit'])) for r in purchases])
            total_qty = sum([r['quantity_purchased'] for r in purchases])
            unique_suppliers = len(set([r['supplier'] for r in purchases]))
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Purchases", total_purchases)
            col2.metric("Total Spent", f"{CURRENCY_SYMBOL}{total_spent:,.2f}")
            col3.metric("Total Quantity", f"{total_qty:,.2f}")
            col4.metric("Suppliers", unique_suppliers)
            
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Purchase Book (CSV)",
                data=csv,
                file_name=f'purchase_book_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
                type="primary"
            )
        else:
            st.info("No purchase records found.")


def show_sales_forecasting(system):
    st.markdown("""
    <div class="main-header">
        <h1>📈 AI-Powered Sales Forecasting</h1>
        <p>Predict future sales with machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    products = supabase.table("products").select("id, name, sku").order("name").execute().data
    
    if products:
        opts = {p['id']: f"{p['name']} ({p['sku']})" for p in products}
        pid = st.selectbox("Select Product", options=list(opts.keys()), format_func=lambda x: opts[x])
        horizon = st.slider("Forecast Horizon (Months)", 1, 12, 3)
        
        if st.button("🔮 Generate AI Forecast", type="primary", use_container_width=True):
            with st.spinner("🤖 Agent analyzing patterns..."):
                forecast = system.decision_engine.forecasting_agent.generate_forecast(pid, horizon)
            
            st.subheader(f"Analysis for {opts[pid]}")
            
            if forecast and not forecast['historical'].empty:
                hist_df = forecast['historical']
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['quantity'], mode='lines+markers', name='Historical Sales', line=dict(color='#667eea', width=3)))
                
                if 'predictions' in forecast:
                    preds = forecast['predictions']
                    last_date = hist_df['date'].max()
                    future_dates = [last_date + relativedelta(months=i+1) for i in range(horizon)]
                    fig.add_trace(go.Scatter(x=future_dates, y=preds, mode='lines+markers', name='Forecasted Sales', line=dict(dash='dash', color='#ff7f0e', width=3)))
            
                fig.update_layout(
                    title="Sales Data Analysis",
                    xaxis_title="Date",
                    yaxis_title="Quantity Sold",
                    hovermode='x unified',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if forecast and 'error' in forecast:
                st.warning(forecast['error'])
            elif forecast:
                preds = forecast['predictions']
                trend = forecast['trend']
                conf = forecast['confidence']
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Trend", trend.upper(), delta=f"↑" if trend == "increasing" else "↓" if trend == "decreasing" else "→")
                c2.metric("Confidence", f"{conf*100:.0f}%")
                c3.metric("Next Month Forecast", f"{preds[0]:.1f} units")
                
                future_dates = [datetime.now() + relativedelta(months=i+1) for i in range(horizon)]
                df = pd.DataFrame({
                    'Date': [d.strftime('%Y-%m') for d in future_dates], 
                    'Predicted Sales': np.round(preds, 2), 
                    'Recommended Stock': np.ceil(preds * REORDER_MULTIPLIER)
                })
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.info(f"🤖 **Agent Recommendation:** Stock {df['Recommended Stock'].iloc[0]:.0f} units for next month.")
    else:
        st.info("No products available.")


def show_analytics_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>📊 AI-Powered Analytics Dashboard</h1>
        <p>Ask the AI anything about your inventory, sales, and purchases</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Chat Assistant
    st.subheader("💬 AI Analytics Assistant")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**🧑 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 AI Assistant:**")
                st.markdown(message['content'])
            st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input("Ask a question:", key="analytics_chat_input", placeholder="e.g., What are my top selling products?")
    with col2:
        ask_button = st.button("🚀 Ask AI", type="primary", use_container_width=True)
    
    if ask_button and user_query:
        with st.spinner("🤖 AI analyzing your inventory data..."):
            st.session_state.chat_history.append({'role': 'user', 'content': user_query})
            ai_response = get_ai_analytics_response(user_query)
            st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
            st.rerun()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    with st.expander("💡 Example Questions You Can Ask"):
        st.markdown("""
        **Sales & Revenue:**
        - What are my top 5 selling products?
        - How much revenue did I make this month?
        - What is my average order value?
        
        **Inventory & Stock:**
        - Which products have low stock?
        - What products need reordering?
        
        **Profit & Margins:**
        - What is my profit margin?
        - Calculate my gross profit
        """)

    st.markdown("---")
    st.subheader("📈 Visual Analytics")
    try:
        # Fetch data
        sales_data = supabase.table("sales_log").select("*, products(name, sku, category)").execute().data
        if sales_data:
            sales_df = pd.DataFrame(sales_data)
            # 🔧 FIXED: Robust ISO8601 datetime parsing
            sales_df['date'] = pd.to_datetime(sales_df['created_at'], format='ISO8601', utc=True)
            sales_df['month'] = sales_df['date'].dt.to_period('M').astype(str)
            sales_df['revenue'] = sales_df['quantity_sold'] * sales_df['price_at_sale']

            # KPIs
            st.markdown("### 💰 Key Performance Indicators")
            c1, c2, c3, c4 = st.columns(4)
            total_revenue = sales_df['revenue'].sum()
            total_sales = sales_df['quantity_sold'].sum()
            avg_order_value = sales_df['revenue'].mean()
            total_transactions = len(sales_df)
            c1.metric("Total Revenue", f"{CURRENCY_SYMBOL}{total_revenue:,.2f}")
            c2.metric("Total Units Sold", f"{total_sales:,.2f}")
            c3.metric("Avg Order Value", f"{CURRENCY_SYMBOL}{avg_order_value:,.2f}")
            c4.metric("Transactions", total_transactions)

            # Charts
            st.markdown("### 📅 Monthly Revenue Trend")
            monthly_revenue = sales_df.groupby('month')['revenue'].sum().reset_index()
            fig1 = px.line(monthly_revenue, x='month', y='revenue', title='Monthly Revenue', markers=True)
            fig1.update_traces(line=dict(color='#667eea', width=3))
            fig1.update_layout(template='plotly_white')
            st.plotly_chart(fig1, use_container_width=True)

            if 'products' in sales_df.columns:
                st.markdown("### 🏆 Top Selling Products")
                # Safe product name extraction
                def safe_get_name(x):
                    try:
                        return x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown'
                    except:
                        return 'Unknown'
                sales_df['product_name'] = sales_df['products'].apply(safe_get_name)
                top_products = sales_df.groupby('product_name')['quantity_sold'].sum().nlargest(10).reset_index()
                fig2 = px.bar(top_products, x='quantity_sold', y='product_name', orientation='h', title='Top 10 Products by Quantity')
                fig2.update_traces(marker_color='#764ba2')
                fig2.update_layout(template='plotly_white')
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No sales data available yet. Start recording sales to see analytics!")
    except Exception as e:
        logger.exception("Visual analytics failed")
        st.error("📉 Unable to load visual analytics. Some data may be inconsistent.")
        st.caption(f"Error: {str(e)}")


def show_system_metrics(system):
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ System Metrics</h1>
        <p>Monitor system health and performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    total_p = supabase.table("products").select("id", count="exact").execute().count
    stock_q = supabase.table("products").select("stock_quantity, price").execute().data
    total_v = sum(p['price'] * float(p['stock_quantity']) for p in stock_q if p.get('price') and p.get('stock_quantity'))
    total_i = sum(float(p['stock_quantity']) for p in stock_q if p.get('stock_quantity'))
    faiss_v = system.indexer.index.ntotal
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Products", total_p)
    c2.metric("Total Items", f"{total_i:,.2f}")
    c3.metric("Inventory Value", f"{CURRENCY_SYMBOL}{total_v:,.2f}")
    c4.metric("Indexed Images", faiss_v)
    
    st.markdown("---")
    st.subheader("🔍 System Health")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Database Connection", "✅ Connected" if supabase else "❌ Disconnected")
        st.metric("FAISS Index", f"✅ {faiss_v} embeddings" if faiss_v > 0 else "⚠️ Empty")
    with col2:
        st.metric("Image Processor", "✅ Active" if system.processor.detector else "❌ Inactive")
        st.metric("AI Embedder", "✅ Active")
    
    if st.button("🔄 Rebuild Search Index", type="primary", use_container_width=True):
        with st.spinner("Rebuilding FAISS index..."):
            if system.rebuild_faiss_index():
                st.success("✅ Index rebuilt successfully!")
                st.rerun()


def show_data_management(system):
    st.markdown("""
    <div class="main-header">
        <h1>💾 Data Management</h1>
        <p>Import and export data</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Sync Historical Data")
    st.markdown("Upload your `SALE DATASET.csv` to populate the database.")
    
    csv_file = st.file_uploader("Upload SALE DATASET.csv", type=["csv"])
    
    if csv_file:
        if st.button("🚀 Start Data Synchronization", type="primary", use_container_width=True):
            df = load_and_prepare_csv(csv_file)
            sync_historical_data(df)
            st.info("Running initial agent analysis...")
            system.decision_engine.run_autonomous_cycle()
            st.success("✅ Initial analysis complete!")
    
    st.divider()
    st.subheader("📤 Export Full Sales History")
    
    if st.button("📥 Download Sales Dataset", type="primary", use_container_width=True):
        with st.spinner("Generating dataset..."):
            try:
                all_sales = supabase.table('sales_log').select('created_at, quantity_sold, price_at_sale, products(sku, name)').execute().data
                
                if all_sales:
                    export_data = [{
                        'DATE': pd.to_datetime(r['created_at']).strftime('%m/%d/%Y'),
                        'PRODUCT ID': r['products']['sku'],
                        'NAME': r['products']['name'],
                        'QUANTITY': r['quantity_sold'],
                        'PRICE': r['price_at_sale'],
                        'QUANTITY TYPE': 'Units',
                        'AMOUNT': float(r['quantity_sold']) * float(r['price_at_sale'])
                    } for r in all_sales if r.get('products')]
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Click to Download",
                        data=csv,
                        file_name='complete_sales_dataset.csv',
                        mime='text/csv',
                        type="primary"
                    )
                else:
                    st.warning("No sales data to export.")
            except Exception as e:
                st.error(f"Failed to generate dataset: {e}")

    
    st.divider()
    st.subheader("📥 Import Purchase History")
    st.markdown("Upload a CSV with purchase records. Must include: DATE, I.NO, HSN, NAME, QTY, BASIC")
    purchase_csv = st.file_uploader("Upload PURCHASE DATASET.csv", type=["csv"], key="purchase_uploader")
    if purchase_csv:
        supplier_default = st.text_input("Default Supplier (used if not in CSV)", value="Imported Supplier", key="supplier_input")
        if st.button("🚀 Import Purchase Data", type="primary", use_container_width=True):
            try:
                df = pd.read_csv(purchase_csv)
                # Normalize column names (strip spaces, handle case)
                df.columns = df.columns.str.strip()
                expected_cols = {'DATE', 'I.NO', 'HSN', 'NAME', 'QTY', 'BASIC'}
                if not expected_cols.issubset(df.columns):
                    missing = expected_cols - set(df.columns)
                    st.error(f"❌ Missing columns: {missing}. Required: {expected_cols}")
                    st.stop()

                # Fetch product name → id mapping
                products = supabase.table("products").select("id, name").execute().data
                name_to_id = {p['name']: p['id'] for p in products}

                purchase_records = []
                skipped = 0
                for _, row in df.iterrows():
                    product_name = str(row['NAME']).strip()
                    pid = name_to_id.get(product_name)
                    if pid is None:
                        st.warning(f"⚠️ Product '{product_name}' not found. Skipping row.")
                        skipped += 1
                        continue

                    # Parse numeric fields safely
                    def safe_float(val, default=0.0):
                        try:
                            return float(val) if pd.notna(val) else default
                        except:
                            return default

                    qty = safe_float(row['QTY'])
                    basic = safe_float(row['BASIC'])
                    sgst = safe_float(row.get('SGST', 0))
                    cgst = safe_float(row.get('CGST', 0))
                    igst = safe_float(row.get('IGST', 0))
                    round_off = safe_float(row.get('ROUND OFF', 0))
                    total = safe_float(row.get('TOTAL', qty * basic + round_off))

                    # Parse date
                    date_val = None
                    if 'DATE' in df.columns:
                        try:
                            date_val = pd.to_datetime(row['DATE'], errors='coerce')
                            if pd.isna(date_val):
                                date_val = datetime.now()
                        except:
                            date_val = datetime.now()
                    else:
                        date_val = datetime.now()

                    purchase_records.append({
                        'product_id': pid,
                        'quantity_purchased': qty,
                        'basic_price': basic,
                        'price_per_unit': basic,
                        'sgst_percent': sgst,
                        'cgst_percent': cgst,
                        'igst_percent': igst,
                        'round_off': round_off,
                        'total_amount': total,
                        'supplier': supplier_default or str(row.get('SUPPLIER', 'Imported')),
                        'invoice_no': str(row['I.NO']),
                        'hsn_code': str(row['HSN']),
                        'created_at': date_val.isoformat()
                    })

                if purchase_records:
                    supabase.table("purchase_log").insert(purchase_records).execute()
                    st.success(f"✅ Imported {len(purchase_records)} purchase records! Skipped: {skipped}")
                    st.rerun()
                else:
                    st.warning("No valid records to import.")

            except Exception as e:
                logger.exception("Purchase import failed")
                st.error(f"❌ Import failed: {e}")
    
    st.divider()
    st.subheader("📥 Export Full Purchase History")
    st.markdown("Download your complete purchase ledger in CSV format.")
    if st.button("📥 Download Purchase Book", type="primary", use_container_width=True):
        with st.spinner("Generating purchase dataset..."):
            try:
                purchases = supabase.table("purchase_log").select("*, products(name, sku)").execute().data
                if purchases:
                    export_data = []
                    for r in purchases:
                        product_name = r['products']['name'] if r.get('products') else 'Unknown'
                        export_data.append({
                            'DATE': pd.to_datetime(r['created_at']).strftime('%d/%m/%Y'),
                            'I.NO': r.get('invoice_no', ''),
                            'HSN': r.get('hsn_code', ''),
                            'NAME': product_name,
                            'QTY': float(r['quantity_purchased']),
                            'BASIC': float(r.get('basic_price', r['price_per_unit'])),
                            'SGST': float(r.get('sgst_percent', 0)),
                            'CGST': float(r.get('cgst_percent', 0)),
                            'IGST': float(r.get('igst_percent', 0)),
                            'ROUND OFF': float(r.get('round_off', 0)),
                            'TOTAL': float(r.get('total_amount', r['quantity_purchased'] * r.get('price_per_unit', 0))),
                            'SUPPLIER': r.get('supplier', '')
                        })
                    df = pd.DataFrame(export_data)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="✅ Click to Download Purchase Book",
                        data=csv,
                        file_name=f'purchase_book_{datetime.now().strftime("%Y%m%d")}.csv',
                        mime='text/csv',
                        key="download_purchase_book"
                    )
                else:
                    st.warning("No purchase records found to export.")
            except Exception as e:
                logger.exception("Failed to export purchase data")
                st.error(f"❌ Export failed: {e}")

# ========== MAIN APPLICATION LOGIC ==========
def show_main_application(system):
    """
    This function contains the main dashboard and app pages,
    and is ONLY called AFTER the user is authenticated.
    """
    with st.sidebar:
        # User Profile Card with Avatar
        st.markdown(f"""
        <div class="user-profile-card">
            <div style="font-size: 4rem; margin-bottom: 0.75rem; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));">
                👤
            </div>
            <h3>{st.session_state.get('user_name', 'User')}</h3>
            <p style="font-size: 0.85rem; opacity: 0.9;">{st.session_state.get('user_email', '')}</p>
            <p style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.2); font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
                🎯 {st.session_state.get('user_role', 'user').upper()}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Logout Button
        if st.button("🚪 LOGOUT", use_container_width=True, key="logout_btn"):
            st.session_state.authenticated = False
            st.rerun()
        
        st.markdown("<hr style='margin: 1.5rem 0; border-color: #cbd5e1;'>", unsafe_allow_html=True)
        
        # AI Agent Status Card
        st.markdown("""
        <div class="agent-status-card">
            <div class="agent-status-indicator">
                <div class="agent-pulse-dot"></div>
                <span>🤖 AI AGENT ACTIVE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Run Agent Cycle Button
        if st.button("🔄 RUN AGENT CYCLE", type="primary", use_container_width=True):
            with st.spinner("🤖 AI Agents analyzing inventory..."):
                actions = system.decision_engine.run_autonomous_cycle()
                st.success(f"✅ Generated {len(actions)} actions")
                st.rerun()
        
        st.markdown("<hr style='margin: 1.5rem 0; border-color: #cbd5e1;'>", unsafe_allow_html=True)
        
        # Navigation Header
        st.markdown('<p class="nav-section-header">📍 NAVIGATION MENU</p>', unsafe_allow_html=True)
    
        # Navigation Options
        nav_options = [
            "🤖 Agent Dashboard",
            "🛍️ Point of Sale",
            "📦 Inventory Management",
            "📥 Purchase Management",
            "📈 Sales Forecasting",
            "📊 Analytics Dashboard",
            "⚙️ System Metrics",
            "💾 Data Management"
        ]
        
        if st.session_state.get('user_role') == 'admin':
            nav_options.insert(1, "👥 User Management")
        
        page = st.sidebar.radio("", nav_options, label_visibility="collapsed")
    
    # Route to appropriate page
    if "Agent Dashboard" in page:
        show_agent_dashboard(system)
    elif "User Management" in page:
        show_user_management()
    elif "Point of Sale" in page:
        show_point_of_sale(system)
    elif "Inventory Management" in page:
        show_inventory_management(system)
    elif "Purchase Management" in page:
        show_purchase_management(system)
    elif "Sales Forecasting" in page:
        show_sales_forecasting(system)
    elif "Analytics Dashboard" in page:
        show_analytics_dashboard()
    elif "System Metrics" in page:
        show_system_metrics(system)
    elif "Data Management" in page:
        show_data_management(system)


def main():
    """
    Main application entry point.
    Handles authentication and routes to the correct view.
    """
    # 1. Load the professional CSS styles for the *entire* app
    load_custom_css()
    
    # 2. Check for Supabase connection
    if not supabase:
        st.error("⚠️ Unable to connect to database. Please check your configuration.")
        st.stop()
    
    # 3. Check authentication status
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # 4. Route to the correct view
    if not st.session_state.authenticated:
        # User is NOT logged in:
        # Show the professional "landing/login" page.
        # The show_login_page() function IS your landing page.
        show_login_page()
    else:
        # User IS logged in:
        # Initialize the system and show the main dashboard.
        system = InventoryManagementSystem()
        show_main_application(system)


# ========== INITIALIZE AND RUN ==========
if __name__ == "__main__":
    try:
        supabase = get_supabase_client()
    except Exception as e:
        supabase = None
        logger.exception(f"Failed to initialize Supabase client: {e}")

    main()