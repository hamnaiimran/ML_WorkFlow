import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.data_loader import load_csv_data, fetch_yahoo_finance_data, get_summary_stats
    from utils.feature_engineering import create_date_features, create_lag_features, create_rolling_features, select_features, apply_pca
    from utils.visualization import plot_missing_values, plot_correlation_matrix, plot_feature_importance, plot_regression_results, plot_confusion_matrix, plot_clusters, plot_elbow_method, plot_train_test_split, plot_stock_data, plot_pca_explained_variance
except ImportError as e:
    st.error(f"Error importing utility modules: {str(e)}")
    st.error("Please ensure all utility files are in the utils directory.")
    st.stop()

# Custom theme and page config
st.set_page_config(
    page_title="Data Workflow Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/data-workflow-intelligence',
        'Report a bug': "https://github.com/yourusername/data-workflow-intelligence/issues",
        'About': "### Data Workflow Intelligence\n\nA professional-grade data analysis and modeling platform."
    }
)

# Enhanced Custom CSS with special effects
st.markdown("""
    <style>
    /* Main app styling with animated gradient */
    div[data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #1a1a1a, #2d2d2d, #1a1a1a, #2d2d2d);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Container styling with glass effect */
    div[data-testid="stAppViewBlockContainer"] {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 16px;
        margin: 1rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Header styling with neon effect */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        margin-bottom: 1.5rem !important;
        text-shadow: 0 0 10px rgba(52, 152, 219, 0.5),
                     0 0 20px rgba(52, 152, 219, 0.3),
                     0 0 30px rgba(52, 152, 219, 0.2) !important;
        position: relative !important;
        display: inline-block !important;
        animation: glow 2s ease-in-out infinite alternate !important;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 10px rgba(52, 152, 219, 0.5),
                         0 0 20px rgba(52, 152, 219, 0.3),
                         0 0 30px rgba(52, 152, 219, 0.2); }
        to { text-shadow: 0 0 20px rgba(52, 152, 219, 0.8),
                       0 0 30px rgba(52, 152, 219, 0.5),
                       0 0 40px rgba(52, 152, 219, 0.3); }
    }
    
    /* Button styling with 3D effect */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3),
                   0 0 0 1px rgba(255,255,255,0.1) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        position: relative !important;
        overflow: hidden !important;
        transform-style: preserve-3d !important;
        transform: perspective(1000px) !important;
    }
    
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) rotateX(10deg) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.4),
                   0 0 0 1px rgba(255,255,255,0.2) !important;
    }
    
    /* Input field styling with focus effect */
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] select,
    div[data-testid="stDateInput"] input {
        background: rgba(51, 51, 51, 0.8) !important;
        border-radius: 12px !important;
        border: 2px solid rgba(68, 68, 68, 0.5) !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
        font-size: 1rem !important;
        color: #ffffff !important;
        backdrop-filter: blur(5px) !important;
        -webkit-backdrop-filter: blur(5px) !important;
    }
    
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stSelectbox"] select:focus,
    div[data-testid="stDateInput"] input:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.3),
                   0 0 15px rgba(52, 152, 219, 0.2) !important;
        outline: none !important;
    }
    
    /* Sidebar styling with glass effect */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(26, 26, 26, 0.9) 0%, rgba(45, 45, 45, 0.9) 100%) !important;
        padding: 2rem 1.5rem !important;
        box-shadow: 4px 0 16px rgba(0,0,0,0.3) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* Dataframe styling with glass effect */
    div[data-testid="stDataFrame"] {
        background: rgba(45, 45, 45, 0.8) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3) !important;
        overflow: hidden !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
    }
    
    /* Alert styling with animation */
    div[data-testid="stAlert"] {
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
        border-left: 4px solid !important;
        background: rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        animation: slideIn 0.5s ease-out !important;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(45, 45, 45, 0.8) !important;
        border-radius: 4px !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        border-radius: 4px !important;
        box-shadow: 0 0 5px rgba(52, 152, 219, 0.5) !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"  # Set default theme to dark

# Theme selector in sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='color: white; font-size: 2rem; margin-bottom: 0.5rem;'>📊 Data Workflow Intelligence</h1>
            <p style='color: rgba(255, 255, 255, 0.8); font-size: 1rem;'>Professional Data Analysis Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Theme selector
    theme = st.selectbox(
        "Select Theme",
        ["Dark", "Light", "Blue", "Green"],
        help="Choose your preferred theme"
    )
    
    # Update theme based on selection
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        st.experimental_rerun()
    
    # Navigation menu
    page = option_menu(
        menu_title=None,
        options=["Data Loading", "Feature Engineering", "Visualization", "Model Training"],
        icons=["cloud-upload", "gear", "graph-up", "robot"],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "rgba(255, 255, 255, 0.1)"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "rgba(255, 255, 255, 0.1)",
                "color": "white"
            },
            "nav-link-selected": {
                "background-color": "#3498db",
                "color": "white",
                "font-weight": "bold"
            },
        }
    )

# Data Loading Page
if page == "Data Loading":
    st.title("📥 Data Loading")
    
    with st.container():
        st.markdown("### 📂 Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your dataset in CSV format"
        )
        
        if uploaded_file is not None:
            with st.spinner('Loading your data...'):
                try:
                    data = load_csv_data(uploaded_file)
                    if data is not None:
                        st.session_state.data = data
                        st.success("✅ Data loaded successfully!")
                        with st.container():
                            st.markdown("### Data Preview")
                            st.dataframe(
                                data.head().style.set_properties(**{
                                    'background-color': '#f8f9fa',
                                    'border': '1px solid #e0e0e0',
                                    'border-radius': '8px'
                                })
                            )
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
    
    with st.container():
        st.markdown("### 📈 Fetch Stock Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input(
                "Enter Stock Ticker",
                "AAPL",
                help="Enter a valid stock ticker symbol"
            )
        with col2:
            start_date = st.date_input(
                "Start Date",
                help="Select the start date for historical data"
            )
        with col3:
            end_date = st.date_input(
                "End Date",
                help="Select the end date for historical data"
            )
        
        update_interval = st.selectbox(
            "Update Interval",
            ["None", "5 minutes", "15 minutes", "30 minutes", "1 hour"],
            help="Select how often to update the stock data"
        )
        
        if st.button("Fetch Stock Data", key="fetch_stock"):
            with st.spinner('Fetching stock data...'):
                try:
                    data = fetch_yahoo_finance_data(ticker, start_date, end_date)
                    if data is not None:
                        st.session_state.data = data
                        st.session_state.last_update = datetime.now()
                        st.success("✅ Stock data fetched successfully!")
                        with st.container():
                            st.markdown("### Stock Data Preview")
                            st.dataframe(
                                data.head().style.set_properties(**{
                                    'background-color': '#f8f9fa',
                                    'border': '1px solid #e0e0e0',
                                    'border-radius': '8px'
                                })
                            )
                        
                        if update_interval != "None":
                            interval_minutes = {
                                "5 minutes": 5,
                                "15 minutes": 15,
                                "30 minutes": 30,
                                "1 hour": 60
                            }[update_interval]
                            st.info(f"🔄 Data will be updated every {update_interval}")
                except Exception as e:
                    st.error(f"❌ Error fetching stock data: {str(e)}")

# Feature Engineering Page
elif page == "Feature Engineering":
    st.title("⚙️ Feature Engineering")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from the Data Loading page.")
    else:
        data = st.session_state.data
        
        with st.container():
            st.markdown("### 📅 Date Features")
            date_column = st.selectbox(
                "Select Date Column",
                data.columns,
                help="Select the column containing date information"
            )
            if st.button("Create Date Features", key="date_features"):
                with st.spinner('Creating date features...'):
                    try:
                        data = create_date_features(data, date_column)
                        st.session_state.data = data
                        st.success("✅ Date features created successfully!")
                    except Exception as e:
                        st.error(f"❌ Error creating date features: {str(e)}")
        
        with st.container():
            st.markdown("### ⏱️ Lag Features")
            lag_column = st.selectbox(
                "Select Column for Lag Features",
                data.columns,
                help="Select the column to create lag features for"
            )
            lag_periods = st.multiselect(
                "Select Lag Periods",
                [1, 2, 3, 4, 5, 6, 7, 14, 30],
                help="Select the lag periods to create"
            )
            if st.button("Create Lag Features", key="lag_features"):
                with st.spinner('Creating lag features...'):
                    try:
                        data = create_lag_features(data, lag_column, lag_periods)
                        st.session_state.data = data
                        st.success("✅ Lag features created successfully!")
                    except Exception as e:
                        st.error(f"❌ Error creating lag features: {str(e)}")
        
        with st.container():
            st.markdown("### 📊 Rolling Features")
            rolling_column = st.selectbox(
                "Select Column for Rolling Features",
                data.columns,
                help="Select the column to create rolling features for"
            )
            windows = st.multiselect(
                "Select Window Sizes",
                [3, 5, 7, 10, 14, 30],
                help="Select the window sizes for rolling calculations"
            )
            if st.button("Create Rolling Features", key="rolling_features"):
                with st.spinner('Creating rolling features...'):
                    try:
                        data = create_rolling_features(data, rolling_column, windows)
                        st.session_state.data = data
                        st.success("✅ Rolling features created successfully!")
                    except Exception as e:
                        st.error(f"❌ Error creating rolling features: {str(e)}")
        
        with st.container():
            st.markdown("### 🔍 Principal Component Analysis")
            pca_columns = st.multiselect(
                "Select Columns for PCA",
                data.select_dtypes(include=[np.number]).columns,
                help="Select numeric columns for PCA"
            )
            n_components = st.number_input(
                "Number of Components",
                min_value=1,
                max_value=len(pca_columns) if pca_columns else 1,
                value=2,
                help="Select the number of principal components"
            )
            if st.button("Apply PCA", key="pca"):
                with st.spinner('Applying PCA...'):
                    try:
                        data, pca = apply_pca(data, pca_columns, n_components)
                        if pca is not None:
                            st.session_state.data = data
                            st.success("✅ PCA applied successfully!")
                            plot_pca_explained_variance(pca)
                    except Exception as e:
                        st.error(f"❌ Error applying PCA: {str(e)}")

# Visualization Page
elif page == "Visualization":
    st.title("📊 Data Visualization")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from the Data Loading page.")
    else:
        data = st.session_state.data
        
        with st.container():
            st.markdown("### 🔍 Missing Values Analysis")
            if st.button("Plot Missing Values", key="missing_values"):
                with st.spinner('Analyzing missing values...'):
                    try:
                        missing_data = data.isnull().sum() / len(data) * 100
                        plot_missing_values(missing_data)
                    except Exception as e:
                        st.error(f"❌ Error analyzing missing values: {str(e)}")
        
        with st.container():
            st.markdown("### 🔗 Correlation Matrix")
            if st.button("Plot Correlation Matrix", key="correlation"):
                with st.spinner('Generating correlation matrix...'):
                    try:
                        plot_correlation_matrix(data)
                    except Exception as e:
                        st.error(f"❌ Error generating correlation matrix: {str(e)}")
        
        with st.container():
            st.markdown("### 📈 Feature Importance")
            target = st.selectbox(
                "Select Target Column",
                data.columns,
                help="Select the target variable for feature importance analysis"
            )
            if st.button("Plot Feature Importance", key="feature_importance"):
                with st.spinner('Calculating feature importance...'):
                    try:
                        selected_features, scores_df = select_features(data, target, data.columns.drop(target))
                        if selected_features:
                            plot_feature_importance(scores_df)
                    except Exception as e:
                        st.error(f"❌ Error calculating feature importance: {str(e)}")

# Model Training Page
elif page == "Model Training":
    st.title("🤖 Model Training")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from the Data Loading page.")
    else:
        data = st.session_state.data
        
        with st.container():
            st.markdown("### 🎯 Model Selection")
            model_type = st.selectbox(
                "Select Model Type",
                ["Regression", "Classification", "Clustering"],
                help="Select the type of model to train"
            )
            
            if model_type in ["Regression", "Classification"]:
                with st.container():
                    st.markdown("### 🔍 Feature Selection")
                    target = st.selectbox(
                        "Select Target Column",
                        data.columns,
                        help="Select the target variable"
                    )
                    features = st.multiselect(
                        "Select Features",
                        data.columns.drop(target),
                        help="Select the features to use for training"
                    )
                
                with st.container():
                    st.markdown("### 🎯 Model Configuration")
                    if model_type == "Regression":
                        model_options = ["Linear Regression", "Random Forest", "Support Vector Machine"]
                    else:
                        model_options = ["Logistic Regression", "Random Forest", "Support Vector Machine"]
                    
                    selected_model = st.selectbox(
                        "Select Model",
                        model_options,
                        help="Select the model to train"
                    )
                    
                    test_size = st.slider(
                        "Test Size (%)",
                        10, 40, 20,
                        help="Select the percentage of data to use for testing"
                    )
                
                if st.button("Train Model", key="train_model"):
                    with st.spinner('Training model...'):
                        try:
                            # Prepare data
                            X = data[features]
                            y = data[target]
                            
                            # Split data
                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
                            
                            # Train model
                            if selected_model == "Linear Regression":
                                model = LinearRegression()
                            elif selected_model == "Logistic Regression":
                                model = LogisticRegression(max_iter=1000)
                            elif selected_model == "Random Forest":
                                if model_type == "Regression":
                                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                                else:
                                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                            elif selected_model == "Support Vector Machine":
                                if model_type == "Regression":
                                    model = SVR()
                                else:
                                    model = SVC()
                            
                            model.fit(X_train, y_train)
                            st.session_state.model = model
                            
                            # Evaluate model
                            y_pred = model.predict(X_test)
                            if model_type == "Regression":
                                mse = mean_squared_error(y_test, y_pred)
                                r2 = r2_score(y_test, y_pred)
                                st.success(f"✅ Model trained successfully! MSE: {mse:.4f}, R²: {r2:.4f}")
                            else:
                                accuracy = accuracy_score(y_test, y_pred)
                                st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.4f}")
                            
                            # Plot feature importance if applicable
                            if hasattr(model, 'coef_'):
                                importance = pd.DataFrame({
                                    'Feature': features,
                                    'Importance': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                                })
                                importance = importance.sort_values('Importance', ascending=False)
                                
                                fig = px.bar(
                                    importance,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title='Feature Importance',
                                    color='Importance',
                                    color_continuous_scale='RdBu'
                                )
                                st.plotly_chart(fig)
                            
                            # Download model
                            model_bytes = joblib.dumps(model)
                            st.download_button(
                                label="📥 Download Model",
                                data=model_bytes,
                                file_name=f"{selected_model.lower().replace(' ', '_')}_model.joblib",
                                mime="application/octet-stream",
                                key="download_model"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error training model: {str(e)}")
            
            elif model_type == "Clustering":
                with st.container():
                    st.markdown("### 🔍 Feature Selection")
                    features = st.multiselect(
                        "Select Features",
                        data.select_dtypes(include=[np.number]).columns,
                        help="Select numeric features for clustering"
                    )
                
                with st.container():
                    st.markdown("### 🎯 Clustering Configuration")
                    algorithm = st.selectbox(
                        "Select Clustering Algorithm",
                        ["K-Means", "DBSCAN", "Agglomerative"],
                        help="Select the clustering algorithm to use"
                    )
                    
                    if algorithm == "K-Means":
                        n_clusters = st.slider(
                            "Number of Clusters",
                            2, 10, 3,
                            help="Select the number of clusters"
                        )
                
                if st.button("Apply Clustering", key="clustering"):
                    with st.spinner('Applying clustering...'):
                        try:
                            X = data[features]
                            
                            if algorithm == "K-Means":
                                from sklearn.cluster import KMeans
                                model = KMeans(n_clusters=n_clusters, random_state=42)
                            elif algorithm == "DBSCAN":
                                from sklearn.cluster import DBSCAN
                                model = DBSCAN(eps=0.5, min_samples=5)
                            else:  # Agglomerative
                                from sklearn.cluster import AgglomerativeClustering
                                model = AgglomerativeClustering(n_clusters=3)
                            
                            clusters = model.fit_predict(X)
                            st.session_state.model = model
                            
                            # Visualize clusters
                            if len(features) >= 2:
                                fig = px.scatter(
                                    data,
                                    x=features[0],
                                    y=features[1],
                                    color=clusters,
                                    title='Cluster Visualization',
                                    color_continuous_scale='viridis'
                                )
                                st.plotly_chart(fig)
                            
                            st.success("✅ Clustering completed successfully!")
                            
                            # Download results
                            results = pd.DataFrame({
                                'Cluster': clusters
                            })
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Clustering Results",
                                data=csv,
                                file_name="clustering_results.csv",
                                mime="text/csv",
                                key="download_clusters"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error applying clustering: {str(e)}") 