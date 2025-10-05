"""
Exoplanet Inference Module
Provides comprehensive inference capabilities for exoplanet classification.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ExoplanetInference:
    """
    🔮 Exoplanet Inference System
    
    Provides comprehensive inference capabilities for exoplanet classification,
    including basic predictions, habitability analysis, and explanations.
    """
    
    def __init__(self):
        """Initialize the inference system"""
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.class_names = ['candidate', 'false_positive', 'confirmed']
        self.feature_names = [
            'period', 'duration', 'depth', 'planet_radius', 'stellar_radius',
            'stellar_temp', 'stellar_mag', 'impact_param', 'transit_snr',
            'num_transits', 'duty_cycle', 'log_period', 'log_planet_radius', 'log_depth'
        ]
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load trained models and preprocessors"""
        try:
            models_path = Path("models/trained_models")
            
            # Load scaler
            scaler_path = models_path / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self.logger.info("✅ Scaler loaded successfully")
            else:
                self.logger.warning("⚠️  Scaler not found, using default")
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            
            # Load label encoder
            encoder_path = models_path / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                self.logger.info("✅ Label encoder loaded successfully")
            else:
                self.logger.warning("⚠️  Label encoder not found")
            
            # Load XGBoost model
            xgb_path = models_path / "xgboost_model.pkl"
            if xgb_path.exists():
                self.models['xgboost'] = joblib.load(xgb_path)
                self.logger.info("✅ XGBoost model loaded successfully")
            
            # Load Random Forest model
            rf_path = models_path / "random_forest_model.pkl"
            if rf_path.exists():
                self.models['random_forest'] = joblib.load(rf_path)
                self.logger.info("✅ Random Forest model loaded successfully")
            
            # Load CNN model
            cnn_path = models_path / "cnn_model.keras"
            if cnn_path.exists():
                try:
                    import tensorflow as tf
                    self.models['cnn'] = tf.keras.models.load_model(cnn_path)
                    self.logger.info("✅ CNN model loaded successfully")
                except Exception as e:
                    self.logger.warning(f"⚠️  CNN model loading failed: {e}")
            
            self.logger.info(f"🔮 Inference system initialized with {len(self.models)} models")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading models: {e}")
    
    def _prepare_features(self, features_dict: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction"""
        try:
            # Create feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                value = features_dict.get(feature_name, 0.0)
                feature_vector.append(float(value))
            
            return np.array(feature_vector).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return np.zeros((1, len(self.feature_names)))
    
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make basic classification prediction
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Prediction results
        """
        try:
            # Prepare features
            feature_vector = self._prepare_features(features)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get predictions from available models
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                if model_name in ['xgboost', 'random_forest']:
                    try:
                        pred_proba = model.predict_proba(feature_vector_scaled)[0]
                        pred_class = model.predict(feature_vector_scaled)[0]
                        
                        predictions[model_name] = self.class_names[pred_class]
                        probabilities[model_name] = {name: float(prob) for name, prob in zip(self.class_names, pred_proba)}
                    except Exception as e:
                        self.logger.warning(f"Error with {model_name}: {e}")
            
            # Ensemble prediction (simple average)
            if len(probabilities) > 0:
                ensemble_proba = np.mean(list(probabilities.values()), axis=0)
                ensemble_class = self.class_names[np.argmax(ensemble_proba)]
                ensemble_confidence = float(np.max(ensemble_proba))
            else:
                ensemble_class = 'unknown'
                ensemble_confidence = 0.0
                ensemble_proba = [0.33, 0.33, 0.34]
            
            return {
                'prediction': ensemble_class,
                'confidence': ensemble_confidence,
                'probabilities': {name: float(prob) for name, prob in zip(self.class_names, ensemble_proba)},
                'model_predictions': predictions,
                'model_probabilities': probabilities
            }
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'probabilities': {name: 0.0 for name in self.class_names},
                'error': str(e)
            }
    
    def predict_with_habitability(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction with habitability analysis
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Prediction results with habitability analysis
        """
        try:
            # Get basic prediction
            prediction = self.predict_single(features)
            
            # Add habitability analysis
            try:
                from features.habitability import HabitabilityCalculator
                habitability_calc = HabitabilityCalculator()
                
                habitability_result = habitability_calc.calculate_habitability_score(
                    orbital_period=features.get('period', 365.25),
                    planet_radius=features.get('planet_radius', 1.0),
                    stellar_temp=features.get('stellar_temp', 5800),
                    stellar_radius=features.get('stellar_radius', 1.0),
                    stellar_mag=features.get('stellar_mag', 12.0)
                )
                
                prediction['habitability'] = habitability_result
                
            except Exception as e:
                self.logger.warning(f"Habitability analysis failed: {e}")
                prediction['habitability'] = {'error': str(e)}
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in habitability prediction: {e}")
            return self.predict_single(features)
    
    def predict_with_explanation(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction with explanation
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Prediction results with explanations
        """
        try:
            # Get basic prediction
            prediction = self.predict_single(features)
            
            # Add explanations
            try:
                from features.explainability import ExoplanetExplainer
                explainer = ExoplanetExplainer()
                
                explanations = explainer.explain_prediction(features)
                prediction['explanations'] = explanations
                
            except Exception as e:
                self.logger.warning(f"Explanation generation failed: {e}")
                prediction['explanations'] = {'error': str(e)}
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in explanation prediction: {e}")
            return self.predict_single(features)
    
    def predict_comprehensive(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make comprehensive prediction with all analyses
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Comprehensive prediction results
        """
        try:
            # Get basic prediction
            prediction = self.predict_single(features)
            
            # Add habitability
            try:
                from features.habitability import HabitabilityCalculator
                habitability_calc = HabitabilityCalculator()
                
                habitability_result = habitability_calc.calculate_habitability_score(
                    orbital_period=features.get('period', 365.25),
                    planet_radius=features.get('planet_radius', 1.0),
                    stellar_temp=features.get('stellar_temp', 5800),
                    stellar_radius=features.get('stellar_radius', 1.0),
                    stellar_mag=features.get('stellar_mag', 12.0)
                )
                
                prediction['habitability'] = habitability_result
                
            except Exception as e:
                self.logger.warning(f"Habitability analysis failed: {e}")
                prediction['habitability'] = {'error': str(e)}
            
            # Add explanations
            try:
                from features.explainability import ExoplanetExplainer
                explainer = ExoplanetExplainer()
                
                explanations = explainer.explain_prediction(features)
                prediction['explanations'] = explanations
                
            except Exception as e:
                self.logger.warning(f"Explanation generation failed: {e}")
                prediction['explanations'] = {'error': str(e)}
            
            # Add anomaly detection
            try:
                from features.anomaly_detection import ExoplanetAnomalyDetector
                anomaly_detector = ExoplanetAnomalyDetector()
                
                # Prepare features for anomaly detection
                sample_data = pd.DataFrame([features])
                X = anomaly_detector.prepare_features(sample_data)
                
                if len(X) > 0:
                    anomaly_detector.fit_isolation_forest(X, contamination=0.1)
                    anomaly_results = anomaly_detector.detect_anomalies(X)
                    
                    if len(anomaly_results) > 0:
                        anomaly_result = anomaly_results[0]
                        prediction['anomaly'] = {
                            'is_anomaly': anomaly_result.is_anomaly,
                            'anomaly_score': anomaly_result.anomaly_score,
                            'anomaly_type': anomaly_result.anomaly_type,
                            'confidence': anomaly_result.confidence
                        }
                
            except Exception as e:
                self.logger.warning(f"Anomaly detection failed: {e}")
                prediction['anomaly'] = {'error': str(e)}
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive prediction: {e}")
            return self.predict_single(features)
    
    def predict_batch(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make batch predictions
        
        Args:
            data: DataFrame with feature columns
            
        Returns:
            List of prediction results
        """
        try:
            results = []
            
            for idx, row in data.iterrows():
                features_dict = row.to_dict()
                prediction = self.predict_single(features_dict)
                prediction['sample_id'] = idx
                results.append(prediction)
            
            self.logger.info(f"✅ Batch prediction completed: {len(results)} samples")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {e}")
            return []

def main():
    """Test the inference system"""
    print("🔮 Testing Exoplanet Inference System")
    print("=" * 40)
    
    # Initialize inference
    inference = ExoplanetInference()
    
    # Test sample
    sample_features = {
        'period': 365.25,
        'duration': 0.1,
        'depth': 0.01,
        'planet_radius': 1.0,
        'stellar_radius': 1.0,
        'stellar_temp': 5800,
        'stellar_mag': 12.0,
        'impact_param': 0.5,
        'transit_snr': 10.0,
        'num_transits': 5,
        'duty_cycle': 0.01,
        'log_period': 5.9,
        'log_planet_radius': 0.0,
        'log_depth': -4.6
    }
    
    # Test different prediction types
    prediction_types = [
        ('Basic Classification', inference.predict_single),
        ('With Habitability', inference.predict_with_habitability),
        ('With Explanation', inference.predict_with_explanation),
        ('Comprehensive', inference.predict_comprehensive)
    ]
    
    for pred_type, pred_func in prediction_types:
        try:
            result = pred_func(sample_features)
            print(f"✅ {pred_type}: {result.get('prediction', 'N/A')}")
        except Exception as e:
            print(f"❌ {pred_type}: Error - {e}")

if __name__ == "__main__":
    main()

