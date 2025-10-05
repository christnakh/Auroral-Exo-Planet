"""
Explainable AI (XAI) Module for Exoplanet Classification
Provides SHAP and LIME explanations for model predictions.
"""

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ExoplanetExplainer:
    """Professional explainability for exoplanet classification models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.explainers = {}
        self.feature_names = [
            'period', 'duration', 'depth', 'planet_radius', 'stellar_radius',
            'stellar_temp', 'stellar_mag', 'impact_param', 'transit_snr',
            'num_transits', 'duty_cycle', 'log_period', 'log_planet_radius', 'log_depth'
        ]
        self.class_names = ['confirmed', 'candidate', 'false_positive']
        self.load_models()
        self.initialize_explainers()
    
    def load_models(self):
        """Load trained models for explanation"""
        try:
            # Load XGBoost model
            xgb_path = self.models_dir / 'xgboost_model.pkl'
            if xgb_path.exists():
                self.models['xgboost'] = joblib.load(xgb_path)
                logger.info("XGBoost model loaded for explanations")
            
            # Load Random Forest model
            rf_path = self.models_dir / 'random_forest_model.pkl'
            if rf_path.exists():
                self.models['random_forest'] = joblib.load(rf_path)
                logger.info("Random Forest model loaded for explanations")
            
            # Load ensemble components
            self.models['ensemble'] = {
                'xgboost': self.models.get('xgboost'),
                'random_forest': self.models.get('random_forest')
            }
            
        except Exception as e:
            logger.error(f"Error loading models for explanation: {e}")
    
    def initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        try:
            # Initialize SHAP explainers
            if 'xgboost' in self.models and self.models['xgboost'] is not None:
                self.explainers['xgboost_shap'] = shap.TreeExplainer(self.models['xgboost'])
                logger.info("XGBoost SHAP explainer initialized")
            
            if 'random_forest' in self.models and self.models['random_forest'] is not None:
                self.explainers['rf_shap'] = shap.TreeExplainer(self.models['random_forest'])
                logger.info("Random Forest SHAP explainer initialized")
            
            # Initialize LIME explainer (using Random Forest as base)
            if 'random_forest' in self.models and self.models['random_forest'] is not None:
                # Create a wrapper function for LIME
                def model_predict_proba(X):
                    return self.models['random_forest'].predict_proba(X)
                
                self.explainers['lime'] = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.random.random((100, len(self.feature_names))),  # Dummy data
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    mode='classification'
                )
                logger.info("LIME explainer initialized")
            
        except Exception as e:
            logger.error(f"Error initializing explainers: {e}")
    
    def explain_prediction_shap(self, X: np.ndarray, model_name: str = 'xgboost') -> Dict[str, Any]:
        """
        Generate SHAP explanations for a prediction
        
        Args:
            X: Input features (1D array)
            model_name: Model to explain ('xgboost' or 'random_forest')
            
        Returns:
            Dictionary with SHAP explanations
        """
        try:
            if f'{model_name}_shap' not in self.explainers:
                raise ValueError(f"SHAP explainer for {model_name} not available")
            
            explainer = self.explainers[f'{model_name}_shap']
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X.reshape(1, -1))
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # Multi-class: get values for predicted class
                predicted_class = self.models[model_name].predict(X.reshape(1, -1))[0]
                class_idx = self.class_names.index(predicted_class)
                shap_values = shap_values[class_idx]
            else:
                # Binary case
                shap_values = shap_values[0]
            
            # Create explanation dictionary
            explanation = {
                'shap_values': shap_values.tolist(),
                'feature_names': self.feature_names,
                'feature_importance': dict(zip(self.feature_names, shap_values)),
                'top_features': self._get_top_features(shap_values, self.feature_names),
                'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0.0,
                'model_name': model_name
            }
            
            logger.info(f"SHAP explanation generated for {model_name}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return {'error': str(e)}
    
    def explain_prediction_lime(self, X: np.ndarray, num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanations for a prediction
        
        Args:
            X: Input features (1D array)
            num_features: Number of top features to explain
            
        Returns:
            Dictionary with LIME explanations
        """
        try:
            if 'lime' not in self.explainers:
                raise ValueError("LIME explainer not available")
            
            explainer = self.explainers['lime']
            
            # Generate LIME explanation
            explanation = explainer.explain_instance(
                X, 
                self.models['random_forest'].predict_proba,
                num_features=num_features
            )
            
            # Extract explanation data
            lime_explanation = {
                'feature_importance': dict(explanation.as_list()),
                'top_features': explanation.as_list()[:num_features],
                'prediction': explanation.predicted_class,
                'confidence': explanation.score,
                'model_name': 'random_forest'
            }
            
            logger.info("LIME explanation generated")
            return lime_explanation
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return {'error': str(e)}
    
    def explain_ensemble_prediction(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Generate comprehensive explanations for ensemble prediction
        
        Args:
            X: Input features (1D array)
            
        Returns:
            Dictionary with comprehensive explanations
        """
        try:
            explanations = {}
            
            # Get predictions from both models
            xgb_pred = self.models['xgboost'].predict_proba(X.reshape(1, -1))[0]
            rf_pred = self.models['random_forest'].predict_proba(X.reshape(1, -1))[0]
            ensemble_pred = (xgb_pred + rf_pred) / 2
            
            # Get SHAP explanations for both models
            if 'xgboost_shap' in self.explainers:
                explanations['xgboost_shap'] = self.explain_prediction_shap(X, 'xgboost')
            
            if 'rf_shap' in self.explainers:
                explanations['random_forest_shap'] = self.explain_prediction_shap(X, 'random_forest')
            
            # Get LIME explanation
            if 'lime' in self.explainers:
                explanations['lime'] = self.explain_prediction_lime(X)
            
            # Combine explanations
            combined_explanation = {
                'predictions': {
                    'xgboost': dict(zip(self.class_names, xgb_pred)),
                    'random_forest': dict(zip(self.class_names, rf_pred)),
                    'ensemble': dict(zip(self.class_names, ensemble_pred))
                },
                'explanations': explanations,
                'feature_names': self.feature_names,
                'class_names': self.class_names
            }
            
            logger.info("Ensemble explanation generated")
            return combined_explanation
            
        except Exception as e:
            logger.error(f"Error generating ensemble explanation: {e}")
            return {'error': str(e)}
    
    def _get_top_features(self, shap_values: np.ndarray, feature_names: List[str], 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top K features by absolute SHAP value"""
        try:
            # Get indices of top features
            top_indices = np.argsort(np.abs(shap_values))[-top_k:][::-1]
            
            top_features = []
            for idx in top_indices:
                top_features.append({
                    'feature': feature_names[idx],
                    'shap_value': float(shap_values[idx]),
                    'importance': abs(shap_values[idx])
                })
            
            return top_features
            
        except Exception as e:
            logger.error(f"Error getting top features: {e}")
            return []
    
    def generate_explanation_report(self, X: np.ndarray, candidate_name: str = "Unknown") -> str:
        """
        Generate a human-readable explanation report
        
        Args:
            X: Input features (1D array)
            candidate_name: Name of the exoplanet candidate
            
        Returns:
            Human-readable explanation report
        """
        try:
            explanation = self.explain_ensemble_prediction(X)
            
            if 'error' in explanation:
                return f"Error generating explanation: {explanation['error']}"
            
            # Get ensemble prediction
            ensemble_pred = explanation['predictions']['ensemble']
            predicted_class = max(ensemble_pred, key=ensemble_pred.get)
            confidence = ensemble_pred[predicted_class]
            
            # Generate report
            report = f"""
🔭 EXOPLANET CLASSIFICATION EXPLANATION REPORT
{'='*50}
Candidate: {candidate_name}
Predicted Class: {predicted_class.title()}
Confidence: {confidence:.1%}

📊 MODEL PREDICTIONS:
"""
            
            for model_name, preds in explanation['predictions'].items():
                report += f"\n{model_name.title()}:"
                for class_name, prob in preds.items():
                    report += f"\n  {class_name}: {prob:.1%}"
            
            # Add feature explanations
            if 'xgboost_shap' in explanation['explanations']:
                xgb_expl = explanation['explanations']['xgboost_shap']
                if 'top_features' in xgb_expl:
                    report += f"\n\n🔍 KEY FEATURES (XGBoost):"
                    for i, feature in enumerate(xgb_expl['top_features'][:5], 1):
                        report += f"\n{i}. {feature['feature']}: {feature['shap_value']:.3f}"
            
            if 'lime' in explanation['explanations']:
                lime_expl = explanation['explanations']['lime']
                if 'top_features' in lime_expl:
                    report += f"\n\n🔍 KEY FEATURES (LIME):"
                    for i, (feature, importance) in enumerate(lime_expl['top_features'][:5], 1):
                        report += f"\n{i}. {feature}: {importance:.3f}"
            
            report += f"\n\n💡 INTERPRETATION:"
            if predicted_class == 'confirmed':
                report += "\nThis candidate shows strong evidence of being a confirmed exoplanet."
            elif predicted_class == 'candidate':
                report += "\nThis candidate requires further observation to confirm planetary nature."
            else:
                report += "\nThis candidate is likely a false positive (stellar variability, instrumental noise, etc.)."
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating explanation report: {e}")
            return f"Error generating report: {str(e)}"

def main():
    """Test the explainability module"""
    explainer = ExoplanetExplainer()
    
    # Test with sample data
    sample_features = np.array([
        10.0, 2.0, 1000.0, 1.0, 1.0, 5800.0, 12.0, 0.5, 10.0, 10,
        0.0083, 1.0, 0.0, 3.0
    ])
    
    print("🔭 Testing Exoplanet Explainer")
    print("="*40)
    
    # Generate explanation
    explanation = explainer.explain_ensemble_prediction(sample_features)
    
    if 'error' not in explanation:
        print("✅ Explanation generated successfully")
        print(f"Predicted class: {max(explanation['predictions']['ensemble'], key=explanation['predictions']['ensemble'].get)}")
    else:
        print(f"❌ Error: {explanation['error']}")
    
    # Generate report
    report = explainer.generate_explanation_report(sample_features, "Test Candidate")
    print("\n" + report)

if __name__ == "__main__":
    main()
