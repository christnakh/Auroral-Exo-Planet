"""
Cross-Mission Generalization Module
Tests model robustness across different space missions (Kepler, TESS, K2).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class CrossMissionValidator:
    """Professional cross-mission validation for exoplanet models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.mission_data = {}
        self.results = {}
        self.load_components()
    
    def load_components(self):
        """Load models and preprocessors"""
        try:
            # Load models
            xgb_path = self.models_dir / 'xgboost_model.pkl'
            rf_path = self.models_dir / 'random_forest_model.pkl'
            
            if xgb_path.exists():
                self.models['xgboost'] = joblib.load(xgb_path)
            if rf_path.exists():
                self.models['random_forest'] = joblib.load(rf_path)
            
            # Load preprocessors
            scaler_path = self.models_dir / 'scaler.pkl'
            encoder_path = self.models_dir / 'label_encoder.pkl'
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
            
            logger.info("Cross-mission validation components loaded")
            
        except Exception as e:
            logger.error(f"Error loading components: {e}")
    
    def prepare_mission_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data by mission for cross-validation
        
        Args:
            df: Combined dataset with mission column
            
        Returns:
            Dictionary with mission-specific datasets
        """
        try:
            mission_data = {}
            
            for mission in df['mission'].unique():
                mission_df = df[df['mission'] == mission].copy()
                
                # Select features
                feature_columns = [
                    'period', 'duration', 'depth', 'planet_radius', 'stellar_radius',
                    'stellar_temp', 'stellar_mag', 'impact_param', 'transit_snr',
                    'num_transits', 'duty_cycle', 'log_period', 'log_planet_radius', 'log_depth'
                ]
                
                # Filter available columns
                available_features = [col for col in feature_columns if col in mission_df.columns]
                X = mission_df[available_features].copy()
                y = mission_df['label']
                
                # Handle missing values
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                mission_data[mission] = {
                    'X': X,
                    'y': y,
                    'features': available_features,
                    'n_samples': len(X)
                }
                
                logger.info(f"Prepared {mission} data: {len(X)} samples")
            
            self.mission_data = mission_data
            return mission_data
            
        except Exception as e:
            logger.error(f"Error preparing mission data: {e}")
            return {}
    
    def train_on_mission_predict_on_others(self, train_mission: str, test_missions: List[str]) -> Dict[str, Dict]:
        """
        Train on one mission, test on others
        
        Args:
            train_mission: Mission to train on
            test_missions: Missions to test on
            
        Returns:
            Dictionary with cross-mission results
        """
        try:
            if train_mission not in self.mission_data:
                raise ValueError(f"Training mission {train_mission} not found")
            
            results = {}
            
            # Get training data
            train_data = self.mission_data[train_mission]
            X_train = train_data['X']
            y_train = train_data['y']
            
            # Encode labels
            y_train_encoded = self.label_encoder.transform(y_train)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train models
            trained_models = {}
            for model_name, model in self.models.items():
                if model_name == 'ensemble':
                    continue
                
                # Create a copy of the model
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train_scaled, y_train_encoded)
                trained_models[model_name] = model_copy
                
                logger.info(f"Trained {model_name} on {train_mission}")
            
            # Test on other missions
            for test_mission in test_missions:
                if test_mission not in self.mission_data:
                    continue
                
                test_data = self.mission_data[test_mission]
                X_test = test_data['X']
                y_test = test_data['y']
                
                # Align features
                X_test = X_test.reindex(columns=train_data['features'], fill_value=0)
                X_test_scaled = self.scaler.transform(X_test)
                y_test_encoded = self.label_encoder.transform(y_test)
                
                mission_results = {}
                
                # Test each model
                for model_name, model in trained_models.items():
                    try:
                        # Make predictions
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test_encoded, y_pred)
                        
                        # Get class names
                        class_names = self.label_encoder.classes_
                        
                        # Classification report
                        report = classification_report(
                            y_test_encoded, y_pred, 
                            target_names=class_names, 
                            output_dict=True
                        )
                        
                        mission_results[model_name] = {
                            'accuracy': accuracy,
                            'classification_report': report,
                            'predictions': y_pred,
                            'probabilities': y_pred_proba,
                            'n_test_samples': len(X_test)
                        }
                        
                        logger.info(f"{model_name} on {test_mission}: {accuracy:.3f} accuracy")
                        
                    except Exception as e:
                        logger.error(f"Error testing {model_name} on {test_mission}: {e}")
                        mission_results[model_name] = {'error': str(e)}
                
                results[f"{train_mission}_to_{test_mission}"] = mission_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-mission validation: {e}")
            return {}
    
    def comprehensive_cross_validation(self) -> Dict[str, Dict]:
        """
        Perform comprehensive cross-mission validation
        
        Returns:
            Dictionary with all cross-mission results
        """
        try:
            logger.info("Starting comprehensive cross-mission validation")
            
            all_results = {}
            missions = list(self.mission_data.keys())
            
            # Test all combinations
            for train_mission in missions:
                test_missions = [m for m in missions if m != train_mission]
                
                if test_missions:
                    results = self.train_on_mission_predict_on_others(train_mission, test_missions)
                    all_results.update(results)
            
            # Calculate summary statistics
            summary = self._calculate_cross_mission_summary(all_results)
            all_results['summary'] = summary
            
            self.results = all_results
            logger.info("Cross-mission validation completed")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive cross-validation: {e}")
            return {}
    
    def _calculate_cross_mission_summary(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for cross-mission results"""
        try:
            summary = {
                'total_experiments': len([k for k in results.keys() if k != 'summary']),
                'model_performance': {},
                'mission_generalization': {},
                'best_combinations': [],
                'worst_combinations': []
            }
            
            # Collect all accuracies
            accuracies = []
            model_accuracies = {}
            mission_combinations = []
            
            for experiment, experiment_results in results.items():
                if experiment == 'summary':
                    continue
                
                for model_name, model_results in experiment_results.items():
                    if 'accuracy' in model_results:
                        acc = model_results['accuracy']
                        accuracies.append(acc)
                        
                        if model_name not in model_accuracies:
                            model_accuracies[model_name] = []
                        model_accuracies[model_name].append(acc)
                        
                        mission_combinations.append({
                            'experiment': experiment,
                            'model': model_name,
                            'accuracy': acc
                        })
            
            # Calculate model performance
            for model_name, accs in model_accuracies.items():
                summary['model_performance'][model_name] = {
                    'mean_accuracy': np.mean(accs),
                    'std_accuracy': np.std(accs),
                    'min_accuracy': np.min(accs),
                    'max_accuracy': np.max(accs),
                    'n_experiments': len(accs)
                }
            
            # Find best and worst combinations
            mission_combinations.sort(key=lambda x: x['accuracy'], reverse=True)
            summary['best_combinations'] = mission_combinations[:3]
            summary['worst_combinations'] = mission_combinations[-3:]
            
            # Overall statistics
            summary['overall_mean_accuracy'] = np.mean(accuracies)
            summary['overall_std_accuracy'] = np.std(accuracies)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating summary: {e}")
            return {}
    
    def generate_cross_mission_report(self) -> str:
        """Generate comprehensive cross-mission validation report"""
        try:
            if not self.results:
                return "No cross-mission results available. Run validation first."
            
            report = f"""
🌌 CROSS-MISSION GENERALIZATION REPORT
{'='*50}

📊 OVERALL PERFORMANCE:
"""
            
            if 'summary' in self.results:
                summary = self.results['summary']
                report += f"Mean Accuracy: {summary['overall_mean_accuracy']:.3f} ± {summary['overall_std_accuracy']:.3f}\n"
                report += f"Total Experiments: {summary['total_experiments']}\n\n"
                
                # Model performance
                report += "🤖 MODEL PERFORMANCE:\n"
                for model_name, perf in summary['model_performance'].items():
                    report += f"\n{model_name.upper()}:\n"
                    report += f"  Mean Accuracy: {perf['mean_accuracy']:.3f} ± {perf['std_accuracy']:.3f}\n"
                    report += f"  Range: {perf['min_accuracy']:.3f} - {perf['max_accuracy']:.3f}\n"
                    report += f"  Experiments: {perf['n_experiments']}\n"
                
                # Best combinations
                report += "\n🏆 BEST COMBINATIONS:\n"
                for i, combo in enumerate(summary['best_combinations'], 1):
                    report += f"{i}. {combo['experiment']} ({combo['model']}): {combo['accuracy']:.3f}\n"
                
                # Worst combinations
                report += "\n⚠️  CHALLENGING COMBINATIONS:\n"
                for i, combo in enumerate(summary['worst_combinations'], 1):
                    report += f"{i}. {combo['experiment']} ({combo['model']}): {combo['accuracy']:.3f}\n"
            
            # Detailed results
            report += "\n📋 DETAILED RESULTS:\n"
            for experiment, experiment_results in self.results.items():
                if experiment == 'summary':
                    continue
                
                report += f"\n{experiment.upper()}:\n"
                for model_name, model_results in experiment_results.items():
                    if 'accuracy' in model_results:
                        report += f"  {model_name}: {model_results['accuracy']:.3f} accuracy\n"
                    elif 'error' in model_results:
                        report += f"  {model_name}: ERROR - {model_results['error']}\n"
            
            report += f"\n💡 INTERPRETATION:\n"
            if 'summary' in self.results:
                mean_acc = self.results['summary']['overall_mean_accuracy']
                if mean_acc > 0.8:
                    report += "✅ Excellent cross-mission generalization! Models are robust across missions.\n"
                elif mean_acc > 0.7:
                    report += "✅ Good cross-mission generalization. Models show reasonable robustness.\n"
                elif mean_acc > 0.6:
                    report += "⚠️  Moderate cross-mission generalization. Some mission-specific tuning may be needed.\n"
                else:
                    report += "❌ Poor cross-mission generalization. Models may be overfitted to training mission.\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating cross-mission report: {e}")
            return f"Error generating report: {str(e)}"

def main():
    """Test cross-mission validation"""
    validator = CrossMissionValidator()
    
    # Load sample data (would normally load from processed data)
    print("🌌 Testing Cross-Mission Validation")
    print("="*40)
    
    # This would normally load the processed data
    # For now, just test the structure
    print("✅ Cross-mission validator initialized")
    print("📊 Ready for cross-mission analysis")

if __name__ == "__main__":
    main()
