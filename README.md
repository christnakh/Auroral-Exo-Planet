# 🔭 Complete Exoplanet ML Analysis System

A comprehensive machine learning system for exoplanet classification, habitability analysis, and scientific discovery using NASA's Kepler, TESS, and K2 datasets.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete model training and analysis
python3 run_all_models.py

# Start the Flask web application
python3 app.py
```

Visit: **http://localhost:5001**

## 📊 System Overview

This system combines multiple machine learning approaches with comprehensive validation and anti-overfitting measures:

- **🤖 Tabular Models**: XGBoost, Random Forest, Ensemble with cross-validation
- **🧠 Deep Learning**: 1D CNN for light curve analysis with regularization
- **🔗 Hybrid Model**: Weighted ensemble combining CNN + tabular features
- **🌍 Scientific Analysis**: Habitability, explanations, anomaly detection
- **🌐 Web Interface**: Complete Flask web application with API

## 🏗️ Project Structure & File Explanations

### **Root Level Files**

#### `app.py` - Main Flask Web Application
**Purpose**: Complete web application with all scientific features
- Serves web pages (home, analysis, batch processing, API docs)
- Provides REST API endpoints for all ML models
- Integrates habitability analysis, explanations, and anomaly detection
- Handles CNN model predictions for light curve analysis

#### `run_all_models.py` - Complete Model Training Script
**Purpose**: Orchestrates the entire ML pipeline from data to deployment
- Runs complete data preprocessing pipeline
- Trains all models (XGBoost, Random Forest, CNN, Hybrid)
- Performs cross-mission validation
- Generates scientific analysis reports
- Creates performance visualizations

#### `run_demo.py` - Demo and Testing Script
**Purpose**: Demonstrates the complete system functionality
- Shows example predictions with all models
- Demonstrates scientific features (habitability, explanations)
- Tests API endpoints

### **features/ - Scientific Analysis Features**

#### `features/data_preprocessing.py` - Data Pipeline
**Purpose**: Handles all data loading, cleaning, and feature engineering
- Loads Kepler KOI, TESS TOI, and K2 datasets
- Standardizes disposition labels across missions
- Creates derived features (duty_cycle, log_period, etc.)
- Handles missing values and infinite values

#### `features/habitability.py` - Habitability Analysis
**Purpose**: Calculates exoplanet habitability scores and related metrics
- Calculates habitable zone boundaries
- Computes equilibrium temperature
- Determines habitability scores (0-1)

#### `features/explainability.py` - Explainable AI
**Purpose**: Provides model explanations using SHAP and LIME
- Generates SHAP explanations for feature importance
- Creates LIME explanations for local interpretability
- Analyzes model decision-making process

#### `features/anomaly_detection.py` - Anomaly Detection
**Purpose**: Identifies unusual exoplanet systems and rare configurations
- Uses Isolation Forest for anomaly detection
- Identifies rare planetary systems
- Detects multi-planet systems

#### `features/cross_mission.py` - Cross-Mission Validation
**Purpose**: Validates model performance across different missions
- Trains on one mission, tests on others
- Measures generalization capability
- Identifies mission-specific biases

#### `features/nasa_integration.py` - NASA API Integration
**Purpose**: Fetches latest exoplanet data from NASA's Exoplanet Archive
- Real-time data updates from NASA API
- Handles API rate limiting and caching
- Integrates with existing datasets

### **models/ - Model Files**

#### `models/cnn_model.py` - CNN for Light Curves
**Purpose**: 1D CNN for analyzing light curve time series data
- Implements 1D CNN architecture for time series
- Processes light curve data with padding/normalization
- Trains on synthetic and real light curve data

#### `models/trained_models/` - Saved Models
**Purpose**: Directory containing all trained model files
- `xgboost_model.pkl` - Trained XGBoost model
- `random_forest_model.pkl` - Trained Random Forest model
- `ensemble_model.pkl` - Trained ensemble model
- `scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Label encoder
- `cnn_model.keras` - Trained CNN model

### **templates/ - Web Interface**

#### `templates/base.html` - Base Template
**Purpose**: Common HTML structure for all pages
- Defines common layout and navigation
- Includes Bootstrap CSS framework
- Provides consistent styling

#### `templates/index.html` - Home Page
**Purpose**: Main landing page with system overview
- Displays system features and capabilities
- Provides navigation to analysis tools
- Shows model performance metrics

#### `templates/analyze.html` - Single Analysis Page
**Purpose**: Interactive form for single exoplanet analysis
- Provides input form for exoplanet parameters
- Shows real-time predictions and explanations
- Displays habitability analysis

#### `templates/batch.html` - Batch Analysis Page
**Purpose**: Interface for batch processing of multiple candidates
- Allows CSV file upload
- Processes multiple exoplanet candidates
- Displays batch results

#### `templates/api_docs.html` - API Documentation
**Purpose**: Complete API reference and testing interface
- Documents all API endpoints
- Provides example requests and responses
- Includes interactive testing interface

### **static/ - Web Assets**

#### `static/css/style.css` - Custom Styling
**Purpose**: Custom CSS for the web application
- Defines custom styling for scientific interface
- Provides responsive design
- Includes dark theme support

#### `static/js/main.js` - JavaScript Functionality
**Purpose**: Interactive features for the web interface
- Handles form submissions and API calls
- Provides real-time updates
- Manages data visualization

## 🤖 Model Architecture

### **Tabular Models**
- **XGBoost**: Gradient boosting with 80.4% accuracy
- **Random Forest**: Ensemble of decision trees with 78.0% accuracy
- **Ensemble**: Weighted combination for 79.8% accuracy

### **Deep Learning Models**
- **1D CNN**: Convolutional neural network for light curve analysis
- **Hybrid Model**: Combines CNN (40%) + XGBoost (30%) + Random Forest (30%)

## 🌐 API Endpoints

### **Web Interface**
- `GET /` - Home page
- `GET /analyze` - Single analysis interface
- `GET /batch` - Batch processing interface
- `GET /api-docs` - API documentation

### **Core API**
- `POST /api/predict` - Basic classification
- `POST /api/analyze` - Complete analysis
- `POST /api/comprehensive` - All models including CNN
- `POST /api/habitability` - Habitability analysis
- `POST /api/explain` - AI explanations
- `POST /api/anomaly` - Anomaly detection
- `POST /api/batch` - Batch processing
- `GET /api/health` - System health check

## 🔬 Scientific Features

### **Habitability Analysis**
- **Habitable Zone**: Based on stellar flux and temperature
- **Equilibrium Temperature**: Planet surface temperature
- **Size Score**: Planet radius suitability (0.5-2.0 Earth radii)
- **Flux Score**: Stellar flux in habitable range
- **Combined Score**: Weighted habitability score (0-1)

### **Explainable AI**
- **SHAP Values**: Global and local feature importance
- **LIME Explanations**: Local interpretable explanations
- **Feature Rankings**: Top contributing features
- **Model Transparency**: Human-readable explanations

### **Anomaly Detection**
- **Isolation Forest**: Unsupervised anomaly detection
- **Rare Systems**: Unusual exoplanet configurations
- **Multi-Planet Systems**: Complex planetary systems
- **Statistical Outliers**: Data-driven anomaly detection

## 🚀 Usage Examples

### **Basic Prediction**
```python
from src.inference import ExoplanetInference

# Initialize inference system
inference = ExoplanetInference()

# Make prediction
features = {
    'period': 365.25,
    'planet_radius': 1.0,
    'stellar_temp': 5800,
    'stellar_radius': 1.0,
    'transit_depth': 0.01,
    'transit_duration': 0.1
}

result = inference.predict_single(features)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### **API Usage**
```bash
# Basic prediction
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "period": 365.25,
    "planet_radius": 1.0,
    "stellar_temp": 5800,
    "stellar_radius": 1.0,
    "transit_depth": 0.01,
    "transit_duration": 0.1
  }'

# Comprehensive analysis
curl -X POST http://localhost:5001/api/comprehensive \
  -H "Content-Type: application/json" \
  -d '{
    "period": 365.25,
    "planet_radius": 1.0,
    "stellar_temp": 5800,
    "stellar_radius": 1.0,
    "transit_depth": 0.01,
    "transit_duration": 0.1,
    "stellar_mag": 12.0
  }'
```

## 📊 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| XGBoost | 80.4% | 0.81 | 0.80 | 0.80 | 92.9% |
| Random Forest | 78.0% | 0.79 | 0.78 | 0.78 | 91.9% |
| Ensemble | 79.8% | 0.80 | 0.80 | 0.80 | 92.8% |
| CNN | 82.1% | 0.83 | 0.82 | 0.82 | 94.2% |
| Hybrid | 83.5% | 0.84 | 0.83 | 0.83 | 95.1% |

## 🛠️ Dependencies

### **Core ML Libraries**
- `scikit-learn` - Machine learning algorithms
- `xgboost` - Gradient boosting
- `tensorflow` - Deep learning framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing

### **Scientific Libraries**
- `shap` - Explainable AI
- `lime` - Local explanations
- `matplotlib` - Visualization
- `seaborn` - Statistical plots

### **Web Framework**
- `flask` - Web application framework
- `flask-cors` - Cross-origin resource sharing

## 🚀 Deployment

### **Development**
```bash
python3 app.py
```

### **Production**
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 app:app

# Using uWSGI
uwsgi --http :5001 --wsgi-file app.py --callable app
```

## 📈 Training Pipeline

### **Complete Training**
```bash
# Run complete model training and analysis
python3 run_all_models.py
```

### **Individual Components**
```bash
# Data preprocessing only
python3 src/data_preprocessing.py

# Tabular model training only
python3 src/model_training.py

# CNN model training only
python3 models/cnn_model.py
```

## 🔬 Scientific Applications

### **Exoplanet Discovery**
- Classify new exoplanet candidates
- Identify confirmed planets vs. false positives
- Rank candidates by likelihood

### **Habitability Assessment**
- Calculate planetary habitability scores
- Identify potentially habitable worlds
- Analyze stellar flux and temperature

### **System Analysis**
- Detect multi-planet systems
- Identify rare planetary configurations
- Analyze stellar variability

### **Mission Planning**
- Cross-mission validation for new missions
- Identify mission-specific biases
- Optimize observation strategies

---

**🔭 Ready to discover new worlds? Start the complete system and begin analyzing exoplanets with state-of-the-art machine learning!**
# Auroracle
# Auroracle-Exo-Planet
# Auroracle-Exo-Planet
# Auroracle-Exo-Planet
# Auroral-Exo-Planet
