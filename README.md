# Data Workflow Intelligence

A professional-grade data analysis and modeling platform built with Streamlit.

## Features

- 📥 **Data Loading**: Upload CSV files or fetch real-time stock data from Yahoo Finance
- ⚙️ **Feature Engineering**: Create date features, lag features, rolling features, and apply PCA
- 📊 **Visualization**: Analyze missing values, correlation matrices, and feature importance
- 🤖 **Model Training**: Train regression, classification, and clustering models

## How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**
   ```bash
   streamlit run app.py
   ```

3. **Access the App**
   - Open your web browser and go to `http://localhost:8501`
   - The app will be available with all features ready to use

## File Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
└── utils/                 # Utility modules
    ├── data_loader.py     # Data loading functions
    ├── feature_engineering.py  # Feature creation functions
    ├── preprocessing.py   # Data preprocessing functions
    ├── model_training.py  # Model training functions
    ├── visualization.py   # Visualization functions
    └── __init__.py        # Package initialization
```

## Deployment

To deploy on Streamlit.io:

1. Push your code to a GitHub repository
2. Connect your repository to Streamlit.io
3. Set the main file path to `app.py`
4. Deploy the app

## Notes

- The app requires Python 3.8 or higher
- All dependencies are listed in `requirements.txt`
- The app uses session state to maintain data between pages
- Real-time stock data updates are supported with configurable intervals

## License

MIT License 