# import streamlit as st
# import pandas as pd
# import numpy as np
# from joblib import load
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from sklearn.preprocessing import MinMaxScaler
# import time
# import threading

# # --- Prometheus Setup ---
# try:
#     from prometheus_client import start_http_server, Gauge, Counter, Histogram
#     PROMETHEUS_AVAILABLE = True
# except ImportError:
#     PROMETHEUS_AVAILABLE = False

# # Try to import the prediction pipeline
# try:
#     from pipeline.prediction_pipeline import generate_features_for_customer
#     PIPELINE_AVAILABLE = True
# except ImportError:
#     PIPELINE_AVAILABLE = False

# # This block initializes metrics ONCE and stores them in session_state
# if PROMETHEUS_AVAILABLE and 'metrics' not in st.session_state:
#     st.session_state.metrics = {
#         "ANALYSIS_LATENCY": Histogram('analysis_latency_seconds', 'Time taken for the entire analysis pipeline'),
#         "ARTIFACTS_LOAD_DURATION": Gauge('artifacts_load_duration_seconds', 'Time taken to load initial models and data'),
#         "APP_ERRORS": Counter('app_errors_total', 'Total number of unexpected errors during analysis'),
#         "FILE_UPLOADS": Counter('file_uploads_total', 'Total number of files uploaded'),
#         "FILE_UPLOAD_SIZE": Histogram('file_upload_size_bytes', 'Distribution of uploaded file sizes'),
#         "ANALYSIS_REQUESTS": Counter('analysis_requests_total', 'Total number of times the analysis button was clicked'),
#         "MODEL_PREDICTION_LATENCY": Histogram('model_prediction_latency_seconds', 'Inference time for each model', ['model']),
#         "PREDICTED_SEGMENT_COUNTS": Counter('predicted_segment_counts_total', 'Count of predictions for each customer segment', ['segment']),
#         "PREDICTION_PROBABILITY": Histogram('prediction_probability_percent', 'Distribution of prediction confidence scores')
#     }
#     # Start the server only once
#     if not st.session_state.get('prometheus_server_started', False):
#         def start_prometheus_server():
#             try:
#                 start_http_server(port=8001, addr='0.0.0.0')
#             except OSError:
#                 pass
#         thread = threading.Thread(target=start_prometheus_server, daemon=True)
#         thread.start()
#         st.session_state.prometheus_server_started = True

# # Helper to safely access metrics
# def get_metric(name):
#     if PROMETHEUS_AVAILABLE and 'metrics' in st.session_state:
#         return st.session_state.metrics.get(name)
#     return None

# # --- Main App ---
# st.set_page_config(
#     page_title="Customer Segment & Market Basket Recommendations",
#     page_icon=":shopping_cart:",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
# st.title("Customer Segment & Market Basket Recommendations")

# @st.cache_resource
# def load_artifacts():
#     start_time = time.time()
#     try:
#         artifacts = {
#             "rf_model": load("artifacts/models/best_rf.joblib"),
#             "xgb_model": load("artifacts/models/best_xgb.joblib"),
#             "lgbm_model": load("artifacts/models/best_lgbm.joblib"),
#             "product_clusters": pd.read_csv("artifacts/data/product_clusters.csv"),
#             "cluster_morphology": pd.read_csv("artifacts/data/cluster_morphology.csv", index_col='customer_clusters'),
#             "customers_tsne": load("artifacts/objects/customers_tsne.joblib"),
#             "products_tsne": load("artifacts/objects/products_tsne.joblib")
#         }
#         duration_metric = get_metric('ARTIFACTS_LOAD_DURATION')
#         if duration_metric:
#             duration_metric.set(time.time() - start_time)
#         return artifacts
#     except FileNotFoundError as e:
#         st.error(f"Artifact file not found: {e}. Please ensure all model and data files are in the 'artifacts' directory.")
#         return None

# artifacts = load_artifacts()
# if artifacts is None:
#     st.stop()

# def create_faceted_radar_charts(df: pd.DataFrame):
#     df_radar = df.copy()
#     df_radar = df_radar.drop(columns=['first_purchase_days', 'last_purchase_days'], axis=1)
#     categories = [col for col in df_radar.columns if col != 'CustomerID']
#     df_radar = df_radar[categories]

#     for col in df_radar.columns:
#         df_radar[col] = pd.to_numeric(df_radar[col], errors='coerce')
    
#     df_radar.fillna(0, inplace=True)

#     scaler = MinMaxScaler()
#     df_scaled = pd.DataFrame(scaler.fit_transform(df_radar), index=df_radar.index, columns=df_radar.columns)
    
#     # Create subplots with adjusted column widths
#     fig = make_subplots(rows=1, 
#                         cols=len(df_scaled), 
#                         specs=[[{'type': 'polar'}] * len(df_scaled)], 
#                         subplot_titles=[f"Cluster {i}" for i in df_scaled.index],
#                         horizontal_spacing=0.1)
#     for i, cluster_index in enumerate(df_scaled.index):
#         fig.add_trace(
#             go.Scatterpolar(r=df_scaled.loc[cluster_index].values, 
#                             theta=categories, 
#                             fill='toself', 
#                             name=f'Cluster {cluster_index}',
#                             hoverinfo='none'
#                             ), 
#                             row=1, 
#                             col=i + 1)
    
#     fig.update_layout(height=400, 
#                       showlegend=False, 
#                       title_text="Customer Segment Profiles",
#                       title_x=0.5,
#                       title_y=0.95,
#                       title_font_size=16,
#                       font=dict(family="DMSans", size=8))
#     fig.update_polars(radialaxis_range=[0, 1])
#     return fig

# # --- UI and Logic ---
# with st.sidebar:
#     st.header("Upload Your Data")
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file:
#     get_metric('FILE_UPLOADS').inc()
#     get_metric('FILE_UPLOAD_SIZE').observe(uploaded_file.size)
    
#     try:
#         raw_df = pd.read_csv(uploaded_file, encoding='latin1')
#         raw_df['CustomerID'] = raw_df['CustomerID'].astype(str)
#         customer_list = sorted(raw_df['CustomerID'].unique())
        
#         with st.sidebar:
#             customer_id = st.selectbox("Select a Customer ID", options=customer_list, key="customer_id_selector")
#             analyze_button = st.button("Analyze Customer", key="analyze_btn")

#         if analyze_button and customer_id:
#             get_metric('ANALYSIS_REQUESTS').inc()
#             analysis_start_time = time.time()
            
#             with st.spinner("Analyzing..."):
#                 try:
#                     customer_df = raw_df[raw_df['CustomerID'] == customer_id].copy()
#                     customer_df = customer_df[~customer_df['InvoiceNo'].astype(str).str.startswith('C')]
#                     customer_df['PriceExt'] = customer_df['Quantity'] * customer_df['UnitPrice']
                    
#                     if not PIPELINE_AVAILABLE:
#                         st.error("Prediction pipeline not available.")
#                         st.stop()

#                     scaled_features = generate_features_for_customer(customer_df, raw_df, artifacts["product_clusters"])
                    
#                     # Model prediction
#                     with get_metric('MODEL_PREDICTION_LATENCY').labels(model="rf").time():
#                         rf_proba = artifacts["rf_model"].predict_proba(scaled_features)
#                     with get_metric('MODEL_PREDICTION_LATENCY').labels(model="xgb").time():
#                         xgb_proba = artifacts["xgb_model"].predict_proba(scaled_features)
#                     with get_metric('MODEL_PREDICTION_LATENCY').labels(model="lgbm").time():
#                         lgbm_proba = artifacts["lgbm_model"].predict_proba(scaled_features)

#                     ensemble_proba = (rf_proba + xgb_proba + lgbm_proba) / 3.0
#                     predicted_segment = np.argmax(ensemble_proba, axis=1)[0]
                    
#                     get_metric('PREDICTED_SEGMENT_COUNTS').labels(segment=str(predicted_segment)).inc()
#                     get_metric('PREDICTION_PROBABILITY').observe(np.max(ensemble_proba) * 100)

#                     # Display results
#                     col1, col2, col3 = st.columns(3)
#                     col1.metric("Predicted Segment", f"Cluster {predicted_segment}", border=True, label_visibility="visible")
#                     col2.metric("Total Spend", f"${customer_df['PriceExt'].sum():.2f}", border=True, label_visibility="visible")
#                     col3.metric("Total Orders", customer_df['InvoiceNo'].nunique(), border=True, label_visibility="visible")
                    
#                     st.plotly_chart(
#                         create_faceted_radar_charts(artifacts["cluster_morphology"]), 
#                         use_container_width=True, 
#                         theme=None,
#                         config=dict(displayModeBar=False),
#                         height=400
#                     )
#                     # Recommendations
#                     morphology = artifacts["cluster_morphology"]
#                     top_cat_col = morphology.loc[predicted_segment].filter(like='cat_').idxmax()
#                     top_cat_id = int(top_cat_col.split('_')[1])
#                     recommendations = artifacts["product_clusters"][artifacts["product_clusters"]['product_clusters'] == top_cat_id]
#                     recommendations = recommendations.rename(columns={
#                         'StockCode': 'Product ID',
#                         'mode_description': 'Product Description',
#                         'median_unit_price': 'Price',
#                         'product_clusters': 'Product Cluster'
#                     })
#                     recommendations = recommendations.drop(columns=['tsne_1', 'tsne_2'], axis=1)
#                     tab1, tab2 = st.tabs(["Recommended Products", "Purchase History"])
#                     with tab1: 
#                         st.dataframe(recommendations)
#                     with tab2: 
#                         st.dataframe(customer_df)
#                     get_metric('ANALYSIS_LATENCY').observe(time.time() - analysis_start_time)

#                 except Exception as e:
#                     get_metric('APP_ERRORS').inc()
#                     st.error(f"An error occurred during analysis: {e}")

#     except Exception as e:
#         st.error(f"Failed to read or process the uploaded CSV file: {e}")

# else:
#     st.info("⬆️ Please upload a CSV file to begin analysis.")

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import time
import threading

# --- Safe Prometheus Imports and Initialization ---
try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Try to import the prediction pipeline
try:
    from pipeline.prediction_pipeline import generate_features_for_customer
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# This function will safely initialize our metrics.
def initialize_metrics():
    """
    Initializes Prometheus metrics safely, preventing crashes on app re-runs.
    """
    # Use a flag in session_state to ensure this block runs only once per session.
    if 'metrics_initialized' not in st.session_state:
        st.session_state.metrics = {}
        try:
            # Attempt to create the metrics. This might fail if a previous
            # process didn't shut down cleanly.
            st.session_state.metrics = {
                "ANALYSIS_LATENCY": Histogram('analysis_latency_seconds', 'Time taken for the entire analysis pipeline'),
                "ARTIFACTS_LOAD_DURATION": Gauge('artifacts_load_duration_seconds', 'Time taken to load initial models and data'),
                "APP_ERRORS": Counter('app_errors_total', 'Total number of unexpected errors during analysis'),
                "FILE_UPLOADS": Counter('file_uploads_total', 'Total number of files uploaded'),
                "FILE_UPLOAD_SIZE": Histogram('file_upload_size_bytes', 'Distribution of uploaded file sizes'),
                "ANALYSIS_REQUESTS": Counter('analysis_requests_total', 'Total number of times the analysis button was clicked'),
                "MODEL_PREDICTION_LATENCY": Histogram('model_prediction_latency_seconds', 'Inference time for each model', ['model']),
                "PREDICTED_SEGMENT_COUNTS": Counter('predicted_segment_counts_total', 'Count of predictions for each customer segment', ['segment']),
                "PREDICTION_PROBABILITY": Histogram('prediction_probability_percent', 'Distribution of prediction confidence scores')
            }
        except ValueError:
            # This error means the metrics are already registered in a global
            # registry from a zombie process. We will log this but allow the
            # app to continue. Metrics for this session may not be recorded.
            print("Prometheus metrics already registered. Session state may have been lost.")
            # We pass silently to prevent the app from crashing.
            pass
        
        # Mark as initialized, regardless of success or failure, to prevent retries.
        st.session_state.metrics_initialized = True

    # Start the Prometheus server in a separate thread if it hasn't been started.
    if PROMETHEUS_AVAILABLE and not st.session_state.get('prometheus_server_started', False):
        try:
            start_http_server(port=8001, addr='0.0.0.0')
            st.session_state.prometheus_server_started = True
            print("Prometheus server started on port 8001.")
        except OSError:
            # This handles the case where the port is already in use by a zombie process.
            print("Prometheus server port 8001 is already in use.")
            st.session_state.prometheus_server_started = True # Mark as "started" to avoid retries.

# Call the initialization function at the start of the script.
if PROMETHEUS_AVAILABLE:
    initialize_metrics()

# Helper function to safely access a metric
def get_metric(name):
    """
    Safely retrieves a metric object. If Prometheus is not available or the
    metric doesn't exist, it returns a dummy object that does nothing.
    This prevents the app from crashing if a metric call is made.
    """
    if PROMETHEUS_AVAILABLE and 'metrics' in st.session_state:
        metric = st.session_state.metrics.get(name)
        if metric:
            return metric
            
    # Return a dummy object that supports all required methods if metric is not available
    class DummyMetric:
        def time(self): return self
        def __enter__(self): pass
        def __exit__(self, *args): pass
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    return DummyMetric()


# --- Main App ---
st.set_page_config(
    page_title="Customer Segment & Market Basket Recommendations",
    page_icon=":shopping_cart:",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Customer Segment & Market Basket Recommendations")

@st.cache_resource
def load_artifacts():
    """Loads all necessary model and data artifacts, and records the duration."""
    start_time = time.time()
    try:
        artifacts = {
            "rf_model": load("artifacts/models/best_rf.joblib"),
            "xgb_model": load("artifacts/models/best_xgb.joblib"),
            "lgbm_model": load("artifacts/models/best_lgbm.joblib"),
            "product_clusters": pd.read_csv("artifacts/data/product_clusters.csv"),
            "cluster_morphology": pd.read_csv("artifacts/data/cluster_morphology.csv", index_col='customer_clusters'),
            "customers_tsne": load("artifacts/objects/customers_tsne.joblib"),
            "products_tsne": load("artifacts/objects/products_tsne.joblib")
        }
        # Safely set the duration metric
        get_metric('ARTIFACTS_LOAD_DURATION').set(time.time() - start_time)
        return artifacts
    except FileNotFoundError as e:
        st.error(f"Artifact file not found: {e}. Please ensure all model and data files are in the 'artifacts' directory.")
        return None

artifacts = load_artifacts()
if artifacts is None:
    st.stop()

def create_faceted_radar_charts(df: pd.DataFrame):
    df_radar = df.copy()
    df_radar = df_radar.drop(columns=['first_purchase_days', 'last_purchase_days'], axis=1)
    categories = [col for col in df_radar.columns if col != 'CustomerID']
    df_radar = df_radar[categories]

    for col in df_radar.columns:
        df_radar[col] = pd.to_numeric(df_radar[col], errors='coerce')
    
    df_radar.fillna(0, inplace=True)

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_radar), index=df_radar.index, columns=df_radar.columns)
    
    fig = make_subplots(rows=1, 
                        cols=len(df_scaled), 
                        specs=[[{'type': 'polar'}] * len(df_scaled)], 
                        subplot_titles=[f"Cluster {i}" for i in df_scaled.index],
                        horizontal_spacing=0.1)
    for i, cluster_index in enumerate(df_scaled.index):
        fig.add_trace(
            go.Scatterpolar(r=df_scaled.loc[cluster_index].values, 
                            theta=categories, 
                            fill='toself', 
                            name=f'Cluster {cluster_index}',
                            hoverinfo='none'
                            ), 
                            row=1, 
                            col=i + 1)
    
    fig.update_layout(height=400, 
                      showlegend=False, 
                      title_text="Customer Segment Profiles",
                      title_x=0.5,
                      title_y=0.95,
                      title_font_size=16,
                      font=dict(family="DMSans", size=8))
    fig.update_polars(radialaxis_range=[0, 1])
    return fig

# --- UI and Logic ---
with st.sidebar:
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    get_metric('FILE_UPLOADS').inc()
    get_metric('FILE_UPLOAD_SIZE').observe(uploaded_file.size)
    
    try:
        raw_df = pd.read_csv(uploaded_file, encoding='latin1')
        raw_df['CustomerID'] = raw_df['CustomerID'].astype(str)
        customer_list = sorted(raw_df['CustomerID'].unique())
        
        with st.sidebar:
            customer_id = st.selectbox("Select a Customer ID", options=customer_list, key="customer_id_selector")
            analyze_button = st.button("Analyze Customer", key="analyze_btn")

        if analyze_button and customer_id:
            get_metric('ANALYSIS_REQUESTS').inc()
            
            with st.spinner("Analyzing..."):
                try:
                    with get_metric('ANALYSIS_LATENCY').time():
                        customer_df = raw_df[raw_df['CustomerID'] == customer_id].copy()
                        customer_df = customer_df[~customer_df['InvoiceNo'].astype(str).str.startswith('C')]
                        customer_df['PriceExt'] = customer_df['Quantity'] * customer_df['UnitPrice']
                        
                        if not PIPELINE_AVAILABLE:
                            st.error("Prediction pipeline not available.")
                            st.stop()

                        scaled_features = generate_features_for_customer(customer_df, raw_df, artifacts["product_clusters"])
                        
                        with get_metric('MODEL_PREDICTION_LATENCY').labels(model="rf").time():
                            rf_proba = artifacts["rf_model"].predict_proba(scaled_features)
                        with get_metric('MODEL_PREDICTION_LATENCY').labels(model="xgb").time():
                            xgb_proba = artifacts["xgb_model"].predict_proba(scaled_features)
                        with get_metric('MODEL_PREDICTION_LATENCY').labels(model="lgbm").time():
                            lgbm_proba = artifacts["lgbm_model"].predict_proba(scaled_features)

                        ensemble_proba = (rf_proba + xgb_proba + lgbm_proba) / 3.0
                        predicted_segment = np.argmax(ensemble_proba, axis=1)[0]
                        
                        get_metric('PREDICTED_SEGMENT_COUNTS').labels(segment=str(predicted_segment)).inc()
                        get_metric('PREDICTION_PROBABILITY').observe(np.max(ensemble_proba) * 100)

                    # Display results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Predicted Segment", f"Cluster {predicted_segment}")
                    col2.metric("Total Spend", f"${customer_df['PriceExt'].sum():.2f}")
                    col3.metric("Total Orders", customer_df['InvoiceNo'].nunique())
                    
                    st.plotly_chart(
                        create_faceted_radar_charts(artifacts["cluster_morphology"]), 
                        use_container_width=True
                    )
                    
                    morphology = artifacts["cluster_morphology"]
                    top_cat_col = morphology.loc[predicted_segment].filter(like='cat_').idxmax()
                    top_cat_id = int(top_cat_col.split('_')[1])
                    recommendations = artifacts["product_clusters"][artifacts["product_clusters"]['product_clusters'] == top_cat_id]
                    recommendations = recommendations.rename(columns={
                        'StockCode': 'Product ID',
                        'mode_description': 'Product Description',
                        'median_unit_price': 'Price',
                        'product_clusters': 'Product Cluster'
                    })
                    recommendations = recommendations.drop(columns=['tsne_1', 'tsne_2'], axis=1)
                    
                    tab1, tab2 = st.tabs(["Recommended Products", "Purchase History"])
                    with tab1: 
                        st.dataframe(recommendations)
                    with tab2: 
                        st.dataframe(customer_df)

                except Exception as e:
                    get_metric('APP_ERRORS').inc()
                    st.error(f"An error occurred during analysis: {e}")

    except Exception as e:
        st.error(f"Failed to read or process the uploaded CSV file: {e}")

else:
    st.info("⬆️ Please upload a CSV file to begin analysis.")
