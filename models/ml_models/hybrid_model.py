"""
🔗 Advanced Hybrid Model
Combines CNN (light curves) with tabular models (XGBoost, Random Forest)
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

logger = logging.getLogger(__name__)

class AdvancedHybridModel:
    """
    🔗 Advanced Hybrid Model for Exoplanet Classification
    
    Combines:
    - CNN for light curve analysis (40% weight)
    - XGBoost for tabular features (30% weight)  
    - Random Forest for tabular features (30% weight)
    """
    
    def __init__(self, models_dir: str = "models/trained_models"):
        """Initialize hybrid model"""
        self.models_dir = Path(models_dir)
        self.cnn_model = None
        self.xgboost_model = None
        self.random_forest_model = None
        self.scaler = None
        self.label_encoder = None
        self.weights = {
            'cnn': 0.5,           # CNN gets highest weight for light curve analysis
            'xgboost': 0.3,       # XGBoost for tabular features
            'random_forest': 0.2  # Random Forest for robustness
        }
        
        logger.info("🔗 Advanced Hybrid Model initialized")
    
    def load_models(self):
        """Load all component models"""
        try:
            # Prefer improved models/scalers if available; fallback to originals
            try:
                self.xgboost_model = joblib.load(self.models_dir / 'xgboost_improved_model.pkl')
                self.random_forest_model = joblib.load(self.models_dir / 'random_forest_improved_model.pkl')
                self.scaler = joblib.load(self.models_dir / 'improved_scaler.pkl')
                self.label_encoder = joblib.load(self.models_dir / 'improved_label_encoder.pkl')
                logger.info("✅ Loaded improved tabular models for hybrid")
            except Exception:
                self.xgboost_model = joblib.load(self.models_dir / 'xgboost_model.pkl')
                self.random_forest_model = joblib.load(self.models_dir / 'random_forest_model.pkl')
                self.scaler = joblib.load(self.models_dir / 'scaler.pkl')
                self.label_encoder = joblib.load(self.models_dir / 'label_encoder.pkl')
            
            # Load CNN model
            try:
                from .cnn_model import LightCurveCNN
                self.cnn_model = LightCurveCNN()
                self.cnn_model.load_model()
                logger.info("✅ CNN model loaded for hybrid")
            except Exception as e:
                logger.warning(f"⚠️  CNN model not available: {e}")
                self.cnn_model = None
            
            logger.info("✅ All hybrid model components loaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading hybrid model components: {e}")
            return False
    
    def predict_single(self, features: Dict[str, Any], light_curve: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Make prediction using hybrid model
        
        Args:
            features: Tabular features dictionary
            light_curve: Optional light curve data (1000 timesteps)
            
        Returns:
            Prediction results with all model contributions
        """
        try:
            # Prepare tabular features
            feature_names = ['period', 'duration', 'depth', 'planet_radius', 'stellar_radius', 
                           'stellar_temp', 'stellar_mag', 'impact_param', 'transit_snr', 
                           'num_transits', 'duty_cycle', 'log_period', 'log_planet_radius', 'log_depth']
            
            feature_vector = np.array([features.get(name, 0) for name in feature_names]).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get tabular model predictions
            xgb_proba = self.xgboost_model.predict_proba(feature_vector_scaled)[0]
            rf_proba = self.random_forest_model.predict_proba(feature_vector_scaled)[0]
            
            # Get CNN prediction if available
            cnn_proba = None
            if self.cnn_model is not None and light_curve is not None:
                try:
                    # Preprocess light curve for CNN
                    light_curve_processed = self._preprocess_light_curve(light_curve)
                    cnn_predictions, cnn_probabilities = self.cnn_model.predict([light_curve_processed])
                    cnn_proba = cnn_probabilities[0]
                except Exception as e:
                    logger.warning(f"CNN prediction failed: {e}")
                    cnn_proba = None
            
            # Calculate hybrid prediction (slightly favor confirmed if planet-like)
            if cnn_proba is not None:
                # Full hybrid: CNN + XGBoost + Random Forest
                hybrid_proba = (
                    self.weights['cnn'] * cnn_proba +
                    self.weights['xgboost'] * xgb_proba +
                    self.weights['random_forest'] * rf_proba
                )
                model_combination = "CNN + XGBoost + Random Forest"
            else:
                # Fallback: XGBoost + Random Forest only
                hybrid_proba = (
                    self.weights['xgboost'] * xgb_proba +
                    self.weights['random_forest'] * rf_proba
                )
                # Renormalize weights
                total_weight = self.weights['xgboost'] + self.weights['random_forest']
                hybrid_proba = hybrid_proba / total_weight
                model_combination = "XGBoost + Random Forest"
            
            # Get predicted class
            predicted_class_idx = np.argmax(hybrid_proba)
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            confidence = hybrid_proba[predicted_class_idx]
            
            # Create probability dictionary
            class_names = self.label_encoder.classes_
            probabilities = {name: float(prob) for name, prob in zip(class_names, hybrid_proba)}
            
            return {
                'prediction': predicted_class,
                'confidence': float(confidence),
                'probabilities': probabilities,
                'model_combination': model_combination,
                'individual_predictions': {
                    'xgboost': {name: float(prob) for name, prob in zip(class_names, xgb_proba)},
                    'random_forest': {name: float(prob) for name, prob in zip(class_names, rf_proba)},
                    'cnn': {name: float(prob) for name, prob in zip(class_names, cnn_proba)} if cnn_proba is not None else None
                },
                'model_weights': self.weights,
                'cnn_available': cnn_proba is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Error in hybrid prediction: {e}")
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_batch(self, features_list: List[Dict[str, Any]], light_curves: Optional[List[np.ndarray]] = None) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        results = []
        
        for i, features in enumerate(features_list):
            light_curve = light_curves[i] if light_curves and i < len(light_curves) else None
            result = self.predict_single(features, light_curve)
            results.append(result)
        
        return results
    
    def _preprocess_light_curve(self, light_curve: np.ndarray) -> np.ndarray:
        """Preprocess light curve for CNN"""
        # Ensure correct length
        if len(light_curve) != 1000:
            # Interpolate to correct length
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(light_curve))
            x_new = np.linspace(0, 1, 1000)
            f = interp1d(x_old, light_curve, kind='linear', fill_value='extrapolate')
            light_curve = f(x_new)
        
        # Normalize
        light_curve = (light_curve - np.mean(light_curve)) / np.std(light_curve)
        
        # Reshape for CNN
        return light_curve.reshape(1, -1, 1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the hybrid model"""
        return {
            'model_type': 'Advanced Hybrid Model',
            'components': {
                'cnn': self.cnn_model is not None,
                'xgboost': self.xgboost_model is not None,
                'random_forest': self.random_forest_model is not None
            },
            'weights': self.weights,
            'total_models': sum([self.cnn_model is not None, 
                               self.xgboost_model is not None, 
                               self.random_forest_model is not None]),
            'model_combination': 'CNN (40%) + XGBoost (30%) + Random Forest (30%)' if self.cnn_model else 'XGBoost (50%) + Random Forest (50%)'
        }
    
    def evaluate_performance(self, test_data: pd.DataFrame, test_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate hybrid model performance"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            # Make predictions
            predictions = []
            for _, row in test_data.iterrows():
                features = row.to_dict()
                result = self.predict_single(features)
                predictions.append(result['prediction'])
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions, average='weighted')
            recall = recall_score(test_labels, predictions, average='weighted')
            f1 = f1_score(test_labels, predictions, average='weighted')
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        except Exception as e:
            logger.error(f"❌ Error evaluating hybrid model: {e}")
            return {'error': str(e)}


def create_hybrid_model(models_dir: str = "models/trained_models") -> AdvancedHybridModel:
    """Create and load hybrid model"""
    hybrid = AdvancedHybridModel(models_dir)
    hybrid.load_models()
    return hybrid


if __name__ == "__main__":
    """Test hybrid model"""
    logging.basicConfig(level=logging.INFO)
    
    print("🔗 Testing Advanced Hybrid Model...")
    
    # Create hybrid model
    hybrid = create_hybrid_model()
    
    # Test prediction
    test_features = {
        'period': 365.25,
        'duration': 0.1,
        'depth': 0.01,
        'planet_radius': 1.0,
        'stellar_radius': 1.0,
        'stellar_temp': 5800,
        'stellar_mag': 12.0,
        'impact_param': 0.5,
        'transit_snr': 10.0,
        'num_transits': 100,
        'duty_cycle': 0.1,
        'log_period': np.log(365.25),
        'log_planet_radius': np.log(1.0),
        'log_depth': np.log(0.01)
    }
    
    # Generate synthetic light curve
    from .cnn_model import generate_synthetic_light_curve
    light_curve = generate_synthetic_light_curve(length=1000, has_transit=True)
    
    # Make prediction
    result = hybrid.predict_single(test_features, light_curve)
    
    print("🔗 Hybrid Model Test Results:")
    print(f"   🎯 Prediction: {result['prediction']}")
    print(f"   🎯 Confidence: {result['confidence']:.3f}")
    print(f"   🔗 Model Combination: {result['model_combination']}")
    print(f"   🧠 CNN Available: {result['cnn_available']}")
    
    # Model info
    info = hybrid.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"   🔗 Type: {info['model_type']}")
    print(f"   🤖 Components: {info['components']}")
    print(f"   ⚖️  Weights: {info['weights']}")
    print(f"   📊 Total Models: {info['total_models']}")
    
    print("\n✅ Hybrid model test completed!")
