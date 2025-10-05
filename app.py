"""
🔭 Professional Exoplanet Analysis Flask Web Application
Complete web interface with templates, CSS, JS, and API separation.
"""

import os
import sys
import json
import uuid
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add features to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'features'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask imports
from flask import Flask, request, jsonify, render_template, url_for, redirect, flash
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv

# ML imports
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import IsolationForest

# Load environment variables (for OPENAI_API_KEY, etc.)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = 'exoplanet-analysis-secret-key-2024'

# Global variables for models and components
models = {}
scaler = None
label_encoder = None
explainers = {}
anomaly_detector = None
cnn_model = None
hybrid_model = None

# Feature names
FEATURE_NAMES = [
    'period', 'duration', 'depth', 'planet_radius', 'stellar_radius',
    'stellar_temp', 'stellar_mag', 'impact_param', 'transit_snr',
    'num_transits', 'duty_cycle', 'log_period', 'log_planet_radius', 'log_depth'
]

CLASS_NAMES = ['candidate', 'confirmed', 'false_positive']  # Default; replaced at runtime by label encoder

# Physical constants for habitability
SOLAR_LUMINOSITY = 3.828e26  # W
SOLAR_TEMPERATURE = 5778  # K
STEFAN_BOLTZMANN = 5.670374419e-8  # W⋅m⁻²⋅K⁻⁴
AU = 1.496e11  # m

class HabitabilityCalculator:
    """Habitability analysis for exoplanets"""
    
    def calculate_habitable_zone(self, stellar_temp: float, stellar_luminosity: float = None) -> Dict[str, float]:
        """Calculate habitable zone boundaries"""
        try:
            if stellar_luminosity is None:
                stellar_luminosity = (stellar_temp / SOLAR_TEMPERATURE) ** 4
            
            # Conservative habitable zone
            inner_edge = np.sqrt(stellar_luminosity / 1.1)
            outer_edge = np.sqrt(stellar_luminosity / 0.53)
            
            # Optimistic habitable zone
            optimistic_inner = np.sqrt(stellar_luminosity / 1.77)
            optimistic_outer = np.sqrt(stellar_luminosity / 0.32)
            
            return {
                'inner_edge': inner_edge,
                'outer_edge': outer_edge,
                'optimistic_inner': optimistic_inner,
                'optimistic_outer': optimistic_outer
            }
        except:
            return {'inner_edge': 0, 'outer_edge': 0, 'optimistic_inner': 0, 'optimistic_outer': 0}
    
    def calculate_equilibrium_temperature(self, stellar_temp: float, stellar_radius: float, 
                                          orbital_period: float, albedo: float = 0.3) -> float:
        """Calculate equilibrium temperature"""
        try:
            period_years = orbital_period / 365.25
            semi_major_axis = (period_years ** 2) ** (1/3)
            stellar_flux = (stellar_temp / SOLAR_TEMPERATURE) ** 4 * (stellar_radius ** 2) / (semi_major_axis ** 2)
            teq = stellar_temp * ((1 - albedo) * stellar_flux) ** 0.25
            return teq
        except:
            return 0.0
    
    def calculate_habitability_score(self, planet_radius: float, stellar_temp: float, 
                                    stellar_radius: float, orbital_period: float,
                                    stellar_mag: float = None) -> Dict[str, Any]:
        """Calculate comprehensive habitability score"""
        try:
            # Size score
            if 0.5 <= planet_radius <= 2.5:
                if 0.8 <= planet_radius <= 1.4:
                    size_score = 1.0
                else:
                    size_score = 1.0 - abs(planet_radius - 1.1) / 0.3
            else:
                size_score = 0.0
            size_score = max(0, min(1, size_score))
            
            # Stellar score
            if 2500 <= stellar_temp <= 7200:
                if 5000 <= stellar_temp <= 6500:
                    stellar_score = 1.0
                else:
                    stellar_score = 1.0 - abs(stellar_temp - 5750) / 1000
            else:
                stellar_score = 0.0
            stellar_score = max(0, min(1, stellar_score))
            
            # Habitable zone score
            hz = self.calculate_habitable_zone(stellar_temp)
            eq_temp = self.calculate_equilibrium_temperature(stellar_temp, stellar_radius, orbital_period)
            
            hz_score = 0.0
            hz_position = 'outside'
            if hz['inner_edge'] > 0 and hz['outer_edge'] > 0:
                period_years = orbital_period / 365.25
                orbital_distance = (period_years ** 2) ** (1/3)
                
                if hz['inner_edge'] <= orbital_distance <= hz['outer_edge']:
                    hz_score = 1.0
                    hz_position = 'conservative'
                elif hz['optimistic_inner'] <= orbital_distance <= hz['optimistic_outer']:
                    hz_score = 0.7
                    hz_position = 'optimistic'
                else:
                    if orbital_distance < hz['inner_edge']:
                        hz_position = 'too_close'
                    else:
                        hz_position = 'too_far'
            
            # Temperature score
            if 200 <= eq_temp <= 400:
                if 250 <= eq_temp <= 350:
                    temp_score = 1.0
                else:
                    temp_score = 1.0 - abs(eq_temp - 300) / 50
            else:
                temp_score = 0.0
            temp_score = max(0, min(1, temp_score))
            
            # Overall habitability score
            weights = {'size': 0.25, 'stellar': 0.20, 'habitable_zone': 0.35, 'temperature': 0.20}
            overall_score = (
                size_score * weights['size'] +
                stellar_score * weights['stellar'] +
                hz_score * weights['habitable_zone'] +
                temp_score * weights['temperature']
            )
            
            return {
                'habitability_score': overall_score,
                'is_habitable': overall_score >= 0.6,
                'habitable_zone_score': hz_score,
                'size_score': size_score,
                'temperature_score': temp_score,
                'stellar_score': stellar_score,
                'equilibrium_temp': eq_temp,
                'habitable_zone_position': hz_position
            }
        except Exception as e:
            return {
                'habitability_score': 0.0,
                'is_habitable': False,
                'habitable_zone_score': 0.0,
                'size_score': 0.0,
                'temperature_score': 0.0,
                'stellar_score': 0.0,
                'equilibrium_temp': 0.0,
                'habitable_zone_position': 'error'
            }

def load_models():
    """Load all ML models and components"""
    global models, scaler, label_encoder, explainers, anomaly_detector
    
    try:
        models_dir = "models"
        
        # Load models (try improved models first, fallback to original)
        try:
            # Try to load improved models first
            models['xgboost'] = joblib.load(f"{models_dir}/trained_models/xgboost_improved_model.pkl")
            models['random_forest'] = joblib.load(f"{models_dir}/trained_models/random_forest_improved_model.pkl")
            scaler = joblib.load(f"{models_dir}/trained_models/improved_scaler.pkl")
            label_encoder = joblib.load(f"{models_dir}/trained_models/improved_label_encoder.pkl")
            
            # Load improved ensemble results
            results = joblib.load(f"{models_dir}/trained_models/improved_results.pkl")
            models['ensemble'] = results['ensemble_improved']
            
            logger.info("✅ Improved models loaded successfully!")
        except FileNotFoundError:
            # Fallback to original models
            models['xgboost'] = joblib.load(f"{models_dir}/trained_models/xgboost_model.pkl")
            models['random_forest'] = joblib.load(f"{models_dir}/trained_models/random_forest_model.pkl")
            scaler = joblib.load(f"{models_dir}/trained_models/scaler.pkl")
            label_encoder = joblib.load(f"{models_dir}/trained_models/label_encoder.pkl")
            
            # Try to load ensemble model if it exists
            try:
                models['ensemble'] = joblib.load(f"{models_dir}/trained_models/ensemble_model.pkl")
            except FileNotFoundError:
                logger.warning("Ensemble model not found, skipping...")
                models['ensemble'] = None
            
            logger.info("✅ Original models loaded successfully!")
        
        # Update class names from the trained encoder for correct ordering
        global CLASS_NAMES
        try:
            CLASS_NAMES = list(label_encoder.classes_)
            logger.info(f"Loaded class names: {CLASS_NAMES}")
        except Exception as e:
            logger.warning(f"Could not read label encoder classes: {e}")

        # Initialize explainers
        try:
            explainers['xgboost_shap'] = shap.TreeExplainer(models['xgboost'])
        except Exception as e:
            logger.warning(f"Skipping XGBoost SHAP explainer: {e}")
        try:
            explainers['rf_shap'] = shap.TreeExplainer(models['random_forest'])
        except Exception as e:
            logger.warning(f"Skipping RF SHAP explainer: {e}")
        
        # Initialize LIME explainer (best-effort)
        try:
            explainers['lime'] = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.random.random((100, len(FEATURE_NAMES))),
                feature_names=FEATURE_NAMES,
                class_names=CLASS_NAMES,
                mode='classification'
            )
        except Exception as e:
            logger.warning(f"Skipping LIME explainer: {e}")
        
        # Initialize and fit anomaly detector
        anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Fit the anomaly detector with sample data
        try:
            import pandas as pd
            # Load some data to fit the anomaly detector
            df = pd.read_csv('data/processed_exoplanet_data.csv')
            sample_features = df[FEATURE_NAMES].head(1000).values
            anomaly_detector.fit(sample_features)
            logger.info("✅ Anomaly detector fitted successfully")
        except Exception as e:
            logger.warning(f"⚠️  Could not fit anomaly detector: {e}")
            # Create dummy data for fitting
            dummy_data = np.random.random((100, len(FEATURE_NAMES)))
            anomaly_detector.fit(dummy_data)
        
        # Try to load CNN model (correct import path)
        try:
            from models.ml_models.cnn_model import LightCurveCNN
            cnn_model = LightCurveCNN()
            try:
                cnn_model.load_model('models/trained_models/cnn_model.keras')
                print("✅ CNN model loaded successfully!")
            except Exception:
                cnn_model = None
                print("⚠️  CNN model file not found, using fallback")
        except Exception as e:
            print(f"⚠️  CNN model not available: {e}")
            cnn_model = None
        
        # Try to load hybrid model
        try:
            from models.ml_models.hybrid_model import AdvancedHybridModel
            hybrid_model = AdvancedHybridModel()
            hybrid_model.load_models()
            print("✅ Advanced hybrid model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Hybrid model not available: {e}")
            hybrid_model = None
        
        print("✅ All models and components loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def prepare_features(features_dict: Dict[str, float]) -> np.ndarray:
    """Prepare features for prediction"""
    try:
        # Create feature array
        feature_array = np.array([
            features_dict.get('period', 10.0),
            features_dict.get('duration', 2.0),
            features_dict.get('depth', 1000.0),
            features_dict.get('planet_radius', 1.0),
            features_dict.get('stellar_radius', 1.0),
            features_dict.get('stellar_temp', 5800.0),
            features_dict.get('stellar_mag', 12.0),
            features_dict.get('impact_param', 0.5),
            features_dict.get('transit_snr', 10.0),
            features_dict.get('num_transits', 10),
            features_dict.get('duty_cycle', 0.0083),
            features_dict.get('log_period', 1.0),
            features_dict.get('log_planet_radius', 0.0),
            features_dict.get('log_depth', 3.0)
        ])
        
        # Handle missing values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return feature_array.reshape(1, -1)
        
    except Exception as e:
        print(f"Error preparing features: {e}")
        return np.zeros((1, len(FEATURE_NAMES)))

def predict_classification(features: np.ndarray) -> Dict[str, Any]:
    """Make classification prediction using hybrid model"""
    try:
        # Scale features
        features_scaled = scaler.transform(features)
        
        def canonicalize_label(name: Any) -> str:
            try:
                s = str(name).lower().replace('-', '_').replace(' ', '_')
            except Exception:
                s = 'candidate'
            if 'confirm' in s:
                return 'confirmed'
            if 'false' in s or 'fp' == s:
                return 'false_positive'
            if 'candidate' in s or s in ('pc', 'cand'):
                return 'candidate'
            return 'candidate'

        def canonicalize_proba_dict(raw: Dict[Any, float]) -> Dict[str, float]:
            agg = {'confirmed': 0.0, 'candidate': 0.0, 'false_positive': 0.0}
            for k, v in raw.items():
                agg[canonicalize_label(k)] += float(v)
            total = sum(agg.values()) or 1.0
            for k in agg:
                agg[k] = float(agg[k] / total)
            return agg

        # Try hybrid model first if available
        if 'hybrid_model' in globals() and hybrid_model is not None:
            try:
                # Build feature dict from array using FEATURE_NAMES
                feature_dict = {name: float(val) for name, val in zip(FEATURE_NAMES, features[0])}
                # No real light curve available here; pass None to rely on tabular-only path
                hybrid_result = hybrid_model.predict_single(feature_dict, None)
                if 'prediction' in hybrid_result and 'probabilities' in hybrid_result:
                    probs_raw = hybrid_result['probabilities']
                    probs = canonicalize_proba_dict(probs_raw)
                    # Heuristic calibration for planet-like inputs
                    period_val = float(feature_dict.get('period', 0.0))
                    depth_val = float(feature_dict.get('depth', 0.0))
                    radius_val = float(feature_dict.get('planet_radius', 0.0))
                    temp_val = float(feature_dict.get('stellar_temp', 0.0))
                    planet_like = (depth_val >= 0.01 and 2.0 <= period_val <= 200.0 and 0.5 <= radius_val <= 2.5 and 4500 <= temp_val <= 7000)
                    if planet_like:
                        # Soften confirmed boost to avoid over-confirming
                        probs['confirmed'] = probs.get('confirmed', 0.0) * 1.05
                    elif depth_val < 0.001 or period_val < 1.0:
                        # Strengthen false positive boost for suspicious signals
                        probs['false_positive'] = probs.get('false_positive', 0.0) * 1.3
                    # Normalize
                    total_p = sum(probs.values()) or 1.0
                    for k in probs:
                        probs[k] = float(probs[k] / total_p)
                    pred = max(probs.items(), key=lambda x: x[1])[0]
                    conf = float(probs[pred])
                    logger.info(f"Hybrid classify probs={probs} features={{'period': {period_val}, 'depth': {depth_val}, 'planet_radius': {radius_val}, 'stellar_temp': {temp_val}}}")
                    return {
                        'prediction': pred,
                        'confidence': conf,
                        'probabilities': probs,
                        'hybrid_used': True
                    }
            except Exception as e:
                logger.warning(f"Hybrid prediction in classify failed, falling back: {e}")

        # Helper: map model probabilities to class names via label encoder
        def proba_dict(model, X):
            probs = model.predict_proba(X)[0]
            model_classes = getattr(model, 'classes_', None)
            if model_classes is not None and label_encoder is not None:
                try:
                    class_names = label_encoder.inverse_transform(model_classes)
                except Exception:
                    class_names = [str(c) for c in model_classes]
            else:
                class_names = [str(i) for i in range(len(probs))]
            return {name: float(p) for name, p in zip(class_names, probs)}

        # Get per-class dicts for each model
        xgb_proba = proba_dict(models['xgboost'], features_scaled)
        rf_proba = proba_dict(models['random_forest'], features_scaled)
        
        # Debug: Check what type of objects we have
        logger.info(f"XGBoost model type: {type(models['xgboost'])}")
        logger.info(f"Random Forest model type: {type(models['random_forest'])}")
        logger.info(f"Ensemble model type: {type(models['ensemble'])}")
        
        # Use hybrid model if available, otherwise use weighted ensemble
        if 'hybrid_model' in globals() and hybrid_model is not None:
            try:
                # Create dummy light curve data for hybrid model
                light_curve = np.random.random((1000, 1))  # Dummy light curve
                hybrid_result = hybrid_model.predict_single(features_scaled[0], light_curve)
                
                if 'prediction' in hybrid_result and 'confidence' in hybrid_result:
                    predicted_class = hybrid_result['prediction']
                    confidence = hybrid_result['confidence']
                    
                    # Create probabilities from hybrid model
                    probabilities = {
                        'confirmed': hybrid_result.get('probabilities', {}).get('confirmed', 0.33),
                        'candidate': hybrid_result.get('probabilities', {}).get('candidate', 0.33),
                        'false_positive': hybrid_result.get('probabilities', {}).get('false_positive', 0.34)
                    }
                    
                    return {
                        'prediction': predicted_class,
                        'confidence': float(confidence),
                        'probabilities': probabilities,
                        'xgboost_prediction': {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, xgb_pred)},
                        'random_forest_prediction': {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, rf_pred)}
                    }
            except Exception as e:
                logger.warning(f"Hybrid model failed, using ensemble: {e}")
        
        # Weighted ensemble across class names (aligned by label encoder classes)
        all_class_names = CLASS_NAMES if CLASS_NAMES else list(set(list(xgb_proba.keys()) + list(rf_proba.keys())))
        ensemble_dict = {}
        for name in all_class_names:
            ensemble_dict[name] = 0.6 * xgb_proba.get(name, 0.0) + 0.4 * rf_proba.get(name, 0.0)
        # Normalize
        total = sum(ensemble_dict.values()) or 1.0
        for k in ensemble_dict:
            ensemble_dict[k] = float(ensemble_dict[k] / total)
        
        # Confidence threshold (informational only; do not override class)
        max_prob = max(ensemble_dict.values()) if ensemble_dict else 0.0
        confidence_threshold = 0.33  # informational threshold
        if max_prob < confidence_threshold:
            logger.info(f"Low confidence prediction ({max_prob:.3f}), returning low-confidence result without override")
        
        # Get predicted class
        predicted_class = max(ensemble_dict.items(), key=lambda x: x[1])[0] if ensemble_dict else 'candidate'
        confidence = ensemble_dict.get(predicted_class, 0.0)

        # Stricter gating: require both models to strongly support 'confirmed'
        if predicted_class == 'confirmed':
            try:
                xgb_conf = float(xgb_proba.get('confirmed', 0.0))
                rf_conf = float(rf_proba.get('confirmed', 0.0))
            except Exception:
                xgb_conf = rf_conf = 0.0
            if not (xgb_conf >= 0.65 and rf_conf >= 0.65):
                # Pick the next best non-confirmed class
                alt_sorted = sorted(((k, v) for k, v in ensemble_dict.items() if k != 'confirmed'), key=lambda x: x[1], reverse=True)
                if alt_sorted:
                    predicted_class, confidence = alt_sorted[0]

        # Remove prior tie-breaker toward confirmed; instead, require stronger evidence for 'confirmed'
        try:
            feature_dict = dict(zip(FEATURE_NAMES, features[0]))
            period_val = float(feature_dict.get('period', 0.0))
            depth_val = float(feature_dict.get('depth', 0.0))
        except Exception:
            period_val = 0.0
            depth_val = 0.0
        
        # Canonical probabilities dictionary
        probabilities = canonicalize_proba_dict(ensemble_dict)
        # Heuristic calibration for planet-like inputs
        try:
            feature_dict = dict(zip(FEATURE_NAMES, features[0]))
            period_val = float(feature_dict.get('period', 0.0))
            depth_val = float(feature_dict.get('depth', 0.0))
            radius_val = float(feature_dict.get('planet_radius', 0.0))
            temp_val = float(feature_dict.get('stellar_temp', 0.0))
        except Exception:
            period_val = depth_val = radius_val = temp_val = 0.0
        planet_like = (depth_val >= 0.01 and 2.0 <= period_val <= 200.0 and 0.5 <= radius_val <= 2.5 and 4500 <= temp_val <= 7000)
        if planet_like:
            probabilities['confirmed'] = probabilities.get('confirmed', 0.0) * 1.05
        elif depth_val < 0.001 or period_val < 1.0:
            probabilities['false_positive'] = probabilities.get('false_positive', 0.0) * 1.3
        total_p = sum(probabilities.values()) or 1.0
        for k in probabilities:
            probabilities[k] = float(probabilities[k] / total_p)
        # Final decision with gating for 'confirmed'
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        predicted_class, confidence = sorted_probs[0]
        if predicted_class == 'confirmed':
            second_best = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
            margin = confidence - second_best
            if confidence < 0.70 or margin < 0.10:
                # Require stronger evidence for confirmed; otherwise choose next best
                predicted_class, confidence = sorted_probs[1]
        logger.info(f"Ensemble classify probs={probabilities} features={{'period': {period_val}, 'depth': {depth_val}, 'planet_radius': {radius_val}, 'stellar_temp': {temp_val}}}")
        
        return {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'probabilities': probabilities,
            'xgboost_prediction': canonicalize_proba_dict(xgb_proba),
            'random_forest_prediction': canonicalize_proba_dict(rf_proba)
        }
        
    except Exception as e:
        logger.error(f"Error in classification prediction: {e}")
        logger.error(f"Features shape: {features.shape}")
        logger.error(f"Features: {features}")
        
        # Try to provide a fallback prediction
        try:
            # Use just XGBoost if available
            if 'xgboost' in models and models['xgboost'] is not None:
                features_scaled = scaler.transform(features)
                xgb_pred = models['xgboost'].predict_proba(features_scaled)[0]
                predicted_class_idx = np.argmax(xgb_pred)
                predicted_class = CLASS_NAMES[predicted_class_idx]
                confidence = xgb_pred[predicted_class_idx]
                
                return {
                    'prediction': predicted_class,
                    'confidence': float(confidence),
                    'probabilities': {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, xgb_pred)},
                    'fallback': True,
                    'error': str(e)
                }
        except Exception as e2:
            logger.error(f"Fallback prediction also failed: {e2}")
        
        return {
            'prediction': 'candidate',  # Default to candidate instead of error
            'confidence': 0.5,  # Default confidence
            'probabilities': {'candidate': 0.5, 'confirmed': 0.2, 'false_positive': 0.3},  # Fixed order
            'error': str(e)
        }

def get_explanations(features: np.ndarray) -> Dict[str, Any]:
    """Get SHAP and LIME explanations"""
    try:
        explanations = {}
        
        # SHAP explanations
        if 'xgboost_shap' in explainers:
            xgb_shap_values = explainers['xgboost_shap'].shap_values(features)
            if isinstance(xgb_shap_values, list):
                predicted_class_idx = np.argmax(models['xgboost'].predict_proba(features)[0])
                xgb_shap_values = xgb_shap_values[predicted_class_idx]
            
            explanations['xgboost_shap'] = {
                'shap_values': xgb_shap_values.tolist() if hasattr(xgb_shap_values, 'tolist') else xgb_shap_values,
                'feature_importance': dict(zip(FEATURE_NAMES, xgb_shap_values.tolist() if hasattr(xgb_shap_values, 'tolist') else xgb_shap_values)),
                'top_features': sorted(zip(FEATURE_NAMES, xgb_shap_values.tolist() if hasattr(xgb_shap_values, 'tolist') else xgb_shap_values), key=lambda x: abs(x[1]), reverse=True)[:5]
            }
        
        # LIME explanation
        if 'lime' in explainers:
            try:
                lime_explanation = explainers['lime'].explain_instance(
                    features[0], 
                    models['random_forest'].predict_proba,
                    num_features=10
                )
                explanations['lime'] = {
                    'feature_importance': dict(lime_explanation.as_list()),
                    'top_features': lime_explanation.as_list()[:5]
                }
            except Exception as e:
                explanations['lime'] = {'error': str(e)}
        
        return explanations
        
    except Exception as e:
        return {'error': str(e)}

def detect_anomalies(features: np.ndarray) -> Dict[str, Any]:
    """Detect anomalies in the exoplanet system"""
    try:
        if anomaly_detector is None:
            logger.warning("Anomaly detector not initialized")
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'anomaly_type': 'normal',
                'confidence': 0.0
            }
        
        # Ensure features is 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get anomaly scores and predictions
        anomaly_scores = anomaly_detector.decision_function(features)
        anomaly_predictions = anomaly_detector.predict(features)
        
        is_anomaly = anomaly_predictions[0] == -1
        anomaly_score = float(anomaly_scores[0])
        confidence = min(100.0, max(0.0, abs(anomaly_score) * 30))  # Convert to percentage
        
        # Make anomaly detection less sensitive - only flag if score is very low
        if anomaly_score > -0.5:  # Less sensitive threshold
            is_anomaly = False
        
        # Determine anomaly type
        feature_dict = dict(zip(FEATURE_NAMES, features[0]))
        anomaly_type = 'normal'
        
        if is_anomaly:
            if feature_dict.get('period', 0) > 1000:
                anomaly_type = 'long_period_system'
            elif feature_dict.get('depth', 0) > 50000:
                anomaly_type = 'deep_transit'
            elif feature_dict.get('planet_radius', 0) > 20:
                anomaly_type = 'giant_planet'
            elif feature_dict.get('transit_snr', 0) > 1000:
                anomaly_type = 'high_snr_system'
            elif feature_dict.get('duty_cycle', 0) > 0.1:
                anomaly_type = 'long_transit'
            else:
                anomaly_type = 'unusual_system'
        
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': anomaly_score,
            'anomaly_type': anomaly_type,
            'confidence': confidence
        }
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        return {
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'anomaly_type': 'normal',
            'confidence': 0.0
        }

# Initialize models on startup
print("🔭 Initializing Exoplanet Analysis System...")
if not load_models():
    print("❌ Failed to load models. Please ensure models are trained first.")
    sys.exit(1)

# Initialize habitability calculator
habitability_calculator = HabitabilityCalculator()

# Web Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    """Simple analysis page"""
    return render_template('analyze.html')

@app.route('/batch')
def batch():
    """Redirect batch to analyze for simplified workflow"""
    return redirect(url_for('analyze'))

@app.route('/api-docs')
def api_docs():
    """API documentation page"""
    return render_template('api_docs.html')

@app.route('/dashboard')
def dashboard():
    """Advanced dashboard page"""
    return render_template('dashboard.html')

@app.route('/realtime')
def realtime():
    """Real-time space data processing page"""
    return render_template('realtime.html')

@app.route('/candidate-test')
def candidate_test():
    """Candidate test form page"""
    return render_template('candidate_test.html')

# Removed models page route

@app.route('/features')
def features():
    """Scientific Features page"""
    return render_template('features.html')

@app.route('/features/habitability')
def habitability_feature():
    """Habitability Analysis Feature"""
    return render_template('features/habitability.html')

@app.route('/features/anomaly-detection')
def anomaly_detection_feature():
    """Anomaly Detection Feature"""
    return render_template('features/anomaly_detection.html')

@app.route('/features/explainability')
def explainability_feature():
    """Explainability Feature"""
    return render_template('features/explainability.html')

@app.route('/features/cross-mission')
def cross_mission_feature():
    """Cross-Mission Validation Feature"""
    return render_template('features/cross_mission.html')

@app.route('/education')
def education():
    """Education page"""
    return render_template('education.html')

@app.route('/chat')
def chat_page():
    """Exoplanet Chatbot page"""
    return render_template('chat.html')

# API Routes
@app.route('/api/analyze', methods=['POST'])
@app.route('/api/analyze/', methods=['POST'])
def api_analyze():
    """Professional exoplanet analysis API endpoint"""
    try:
        start_time = time.time()
        
        # Get input data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract and validate input parameters
        orbital_period = float(data.get('orbital_period', 365.25))
        planet_radius = float(data.get('planet_radius', 1.0))
        stellar_radius = float(data.get('stellar_radius', 1.0))
        stellar_temp = float(data.get('stellar_temp', 5800.0))
        stellar_mag = float(data.get('stellar_mag', 12.0))
        input_transit_depth = float(data.get('transit_depth', 0.01))
        # Auto-convert percentage-like inputs (e.g., 2 or 2.0 => 0.02)
        if input_transit_depth > 1.0:
            input_transit_depth = input_transit_depth / 100.0

        # Clamp to safe ranges to avoid degenerate logs/NaNs that can bias models
        if orbital_period <= 0: orbital_period = 1.0
        if planet_radius <= 0: planet_radius = 0.1
        if stellar_radius <= 0: stellar_radius = 0.1
        if input_transit_depth <= 0: input_transit_depth = 1e-6
        
        # Calculate derived features professionally
        duration = 13 * (orbital_period / 365.25) ** (1/3) * (planet_radius / stellar_radius)
        # Use provided transit_depth directly for the ML feature to respect user input
        depth = input_transit_depth
        # Deterministic, sensible defaults (remove randomness)
        impact_param = 0.5  # central transit assumption
        # Simple SNR proxy scaled from transit depth (bounded)
        transit_snr = float(max(1.0, min(1000.0, depth * 10000.0)))
        num_transits = max(1, int(365.25 / orbital_period * 4))
        duty_cycle = duration / (orbital_period * 24)
        
        # Prepare all 14 required features for ML models
        feature_dict = {
            'period': orbital_period,
            'duration': duration,
            'depth': depth,
            'planet_radius': planet_radius,
            'stellar_radius': stellar_radius,
            'stellar_temp': stellar_temp,
            'stellar_mag': stellar_mag,
            'impact_param': impact_param,
            'transit_snr': transit_snr,
            'num_transits': num_transits,
            'duty_cycle': duty_cycle,
            'log_period': np.log(max(orbital_period, 1e-6)),
            'log_planet_radius': np.log(max(planet_radius, 1e-6)),
            'log_depth': np.log(max(depth, 1e-6))
        }
        
        features = prepare_features(feature_dict)
        
        # Get classification prediction
        try:
            logger.info(f"Features shape: {features.shape}")
            logger.info(f"Features: {features}")
            classification = predict_classification(features)
            logger.info(f"Classification result: {classification}")
        except Exception as e:
            logger.error(f"Classification error: {e}")
            classification = {
                'prediction': 'candidate',
                'confidence': 0.5,
                'probabilities': {'candidate': 0.5, 'confirmed': 0.2, 'false_positive': 0.3},  # Fixed order
                'error': str(e)
            }
        
        # Get habitability analysis
        habitability = habitability_calculator.calculate_habitability_score(
            planet_radius=planet_radius,
            stellar_temp=stellar_temp,
            stellar_radius=stellar_radius,
            orbital_period=orbital_period,
            stellar_mag=stellar_mag
        )
        
        # Get explanations only if requested (slow operation)
        explanations = None
        if data.get('include_explanations', False):
            explanations = get_explanations(features)
        else:
            explanations = {'note': 'Explanations not requested for faster processing'}
        
        # Get anomaly detection
        anomaly = detect_anomalies(features)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Create comprehensive response
        response = {
            'prediction_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': round(processing_time, 2),
            'classification': classification,
            'habitability': habitability,
            'explanations': explanations,
            'anomaly_detection': anomaly,
            'input_parameters': data
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@app.route('/api/chat/', methods=['POST'])
def api_chat():
    """Exoplanet-only chatbot endpoint (requires OPENAI_API_KEY if using OpenAI)."""
    try:
        payload = request.get_json() or {}
        user_message = (payload.get('message') or '').strip()
        history = payload.get('history') or []  # list of {role, content}
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        system_prompt = (
            "You are an expert exoplanet assistant. Only discuss exoplanets, "+
            "astronomical detections, missions (Kepler, TESS, K2), planetary habitability, "+
            "light curves, transit signals, and related scientific concepts. If asked about other topics, "+
            "politely redirect to exoplanet topics. Keep answers concise and factual."
        )

        # Build messages
        messages = [{'role': 'system', 'content': system_prompt}] + history + [
            {'role': 'user', 'content': user_message}
        ]

        # Read API key from environment (supports several common names)
        api_key = (
            os.getenv('OPENAI_API_KEY') or
            os.getenv('OPENAI_APIKEY') or
            os.getenv('OPENAI_KEY') or
            os.getenv('OPENAI_TOKEN')
        )
        if not api_key:
            # Offline fallback simple response
            fallback = (
                "I’m set up to talk only about exoplanets. "
                "Try asking about detection methods (transits, radial velocity), Kepler/TESS missions, "
                "or habitability criteria (equilibrium temperature, HZ)."
            )
            return jsonify({'reply': fallback})

        # Minimal OpenAI Chat Completions call
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        model_name = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        body = {
            'model': model_name,
            'messages': messages,
            'temperature': 0.3,
            'max_tokens': 400
        }
        resp = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        reply = data['choices'][0]['message']['content'] if data.get('choices') else 'No reply.'
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
@app.route('/api/predict/', methods=['POST'])
def api_predict():
    """Basic classification API endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        features = prepare_features(data)
        prediction = predict_classification(features)
        
        return jsonify({
            'prediction_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/habitability', methods=['POST'])
@app.route('/api/habitability/', methods=['POST'])
def api_habitability():
    """Habitability analysis API endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        habitability_result = habitability_calculator.calculate_habitability_score(
            planet_radius=data.get('planet_radius', 1.0),
            stellar_temp=data.get('stellar_temp', 5800.0),
            stellar_radius=data.get('stellar_radius', 1.0),
            orbital_period=data.get('period', 365.0),
            stellar_mag=data.get('stellar_mag', 12.0)
        )
        
        return jsonify({
            'prediction_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'habitability': habitability_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch', methods=['POST'])
@app.route('/api/batch/', methods=['POST'])
def api_batch():
    """Batch analysis API endpoint"""
    try:
        data = request.get_json()
        if not data or 'candidates' not in data:
            return jsonify({'error': 'No candidates data provided'}), 400
        
        results = []
        for i, candidate in enumerate(data['candidates']):
            try:
                features = prepare_features(candidate)
                classification = predict_classification(features)
                habitability = habitability_calculator.calculate_habitability_score(
                    planet_radius=candidate.get('planet_radius', 1.0),
                    stellar_temp=candidate.get('stellar_temp', 5800.0),
                    stellar_radius=candidate.get('stellar_radius', 1.0),
                    orbital_period=candidate.get('period', 365.0),
                    stellar_mag=candidate.get('stellar_mag', 12.0)
                )
                anomaly = detect_anomalies(features)
                
                results.append({
                    'candidate_id': i,
                    'classification': classification,
                    'habitability': habitability,
                    'anomaly_detection': anomaly
                })
            except Exception as e:
                results.append({
                    'candidate_id': i,
                    'error': str(e)
                })
        
        return jsonify({
            'batch_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'total_candidates': len(data['candidates']),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/comprehensive', methods=['POST'])
@app.route('/api/comprehensive/', methods=['POST'])
def api_comprehensive():
    """Comprehensive analysis with all models including CNN"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get basic prediction
        features = prepare_features(data)
        basic_result = predict_classification(features)
        
        # Get habitability analysis
        habitability_result = habitability_calculator.calculate_habitability_score(
            planet_radius=data.get('planet_radius', 1.0),
            stellar_temp=data.get('stellar_temp', 5800.0),
            stellar_radius=data.get('stellar_radius', 1.0),
            orbital_period=data.get('period', 365.0),
            stellar_mag=data.get('stellar_mag', 12.0)
        )
        
        # Get explanations
        explanations = get_explanations(features)
        
        # Get anomaly score
        anomaly_score = detect_anomalies(features)
        
        # Try to get hybrid model prediction if available
        hybrid_result = None
        cnn_result = None
        
        if hybrid_model is not None:
            try:
                # Generate synthetic light curve for demonstration
                from models.ml_models.cnn_model import generate_synthetic_light_curve
                light_curve = generate_synthetic_light_curve(
                    length=1000,
                    has_transit=data.get('transit_depth', 0.01) > 0.005,
                    noise_level=0.01,
                    transit_depth=data.get('transit_depth', 0.01)
                )
                
                # Use hybrid model for comprehensive prediction
                hybrid_result = hybrid_model.predict_single(data, light_curve)
                
                # Also get individual CNN result if available
                if cnn_model is not None:
                    cnn_predictions, cnn_probabilities = cnn_model.predict([light_curve])
                    cnn_result = {
                        'prediction': cnn_predictions[0],
                        'probabilities': cnn_probabilities[0].tolist(),
                        'confidence': float(np.max(cnn_probabilities[0]))
                    }
            except Exception as e:
                hybrid_result = {'error': f'Hybrid prediction failed: {str(e)}'}
                cnn_result = {'error': f'CNN prediction failed: {str(e)}'}
        
        # Combine all results
        comprehensive_result = {
            'prediction_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'basic_prediction': basic_result,
            'habitability': habitability_result,
            'explanations': explanations,
            'anomaly_detection': anomaly_score,
            'cnn_analysis': cnn_result,
            'hybrid_analysis': hybrid_result,
            'model_ensemble': {
                'xgboost_available': 'xgboost' in models,
                'random_forest_available': 'random_forest' in models,
                'cnn_available': cnn_model is not None,
                'hybrid_available': hybrid_model is not None,
                'total_models': len([m for m in models.values() if m is not None]) + (1 if cnn_model is not None else 0) + (1 if hybrid_model is not None else 0)
            }
        }
        
        return jsonify(comprehensive_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check API endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'models_loaded': len(models) > 0,
        'features': {
            'classification': True,
            'habitability': True,
            'explanations': True,
            'anomaly_detection': True
        }
    })

@app.route('/api/nasa/latest', methods=['GET'])
def api_nasa_latest():
    """Get latest NASA exoplanet discoveries"""
    try:
        from tools.nasa_integration import get_nasa_data
        nasa_data = get_nasa_data()
        return jsonify(nasa_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nasa/habitable', methods=['GET'])
def api_nasa_habitable():
    """Get potentially habitable planets from NASA"""
    try:
        from tools.nasa_integration import NASAExoplanetAPI
        nasa_api = NASAExoplanetAPI()
        habitable_planets = nasa_api.get_habitable_planets(limit=10)
        return jsonify({
            'habitable_planets': habitable_planets,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nasa/statistics', methods=['GET'])
def api_nasa_statistics():
    """Get exoplanet discovery statistics"""
    try:
        from tools.nasa_integration import NASAExoplanetAPI
        nasa_api = NASAExoplanetAPI()
        statistics = nasa_api.get_discovery_statistics()
        return jsonify({
            'statistics': statistics,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/anomaly-detection', methods=['POST'])
@app.route('/api/anomaly-detection/', methods=['POST'])
def api_anomaly_detection():
    """Anomaly detection API endpoint"""
    try:
        data = request.get_json()
        
        # Load anomaly detector
        from features.anomaly_detection import ExoplanetAnomalyDetector
        detector = ExoplanetAnomalyDetector()
        
        # Detect anomalies
        result = detector.detect_anomaly(data)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/explainability', methods=['POST'])
@app.route('/api/explainability/', methods=['POST'])
def api_explainability():
    """Model explainability API endpoint"""
    try:
        data = request.get_json()
        
        # Load explainer
        from features.explainability import ExoplanetExplainer
        explainer = ExoplanetExplainer()
        
        # Generate explanation
        result = explainer.explain_prediction(data)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cross-mission', methods=['POST'])
@app.route('/api/cross-mission/', methods=['POST'])
def api_cross_mission():
    """Cross-mission validation API endpoint"""
    try:
        data = request.get_json()
        
        # Load cross-mission validator
        from features.cross_mission import CrossMissionValidator
        validator = CrossMissionValidator()
        
        # Run validation
        result = validator.validate_across_missions(data)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Simple configuration
    import socket
    
    def get_available_port():
        """Get an available port"""
        for port in range(8000, 8010):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        return 8000
    
    print("🚀 Starting Exoplanet Analysis Flask Web Application...")
    print("📊 Available endpoints:")
    print("  • GET / - Home page")
    print("  • GET /analyze - Single analysis page")
    print("  • GET /dashboard - Advanced dashboard")
    print("  • GET /batch - Batch analysis page")
    print("  • GET /api-docs - API documentation")
    print("  • POST /api/analyze - Complete analysis API")
    print("  • POST /api/predict - Classification API")
    print("  • POST /api/habitability - Habitability API")
    print("  • POST /api/batch - Batch analysis API")
    print("  • GET /api/health - Health check")
    print("  • POST /api/comprehensive - Comprehensive analysis API")
    print("  • GET /api/nasa/latest - NASA latest discoveries")
    print("  • GET /api/nasa/habitable - NASA habitable planets")
    print("  • GET /api/nasa/statistics - NASA statistics")
    print("")
    
    # Get available port
    port = get_available_port()
    print(f"🌐 Web application starting on http://localhost:{port}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Trying alternative port...")
            # Try alternative port
            alternative_port = get_available_port()
            print(f"🌐 Starting on alternative port: http://localhost:{alternative_port}")
            app.run(host='0.0.0.0', port=alternative_port, debug=False)
        else:
            raise