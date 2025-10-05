"""
Time-Series Anomaly Detection Module
Detects unusual transits and rare exoplanet systems using autoencoders and isolation forests.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    confidence: float
    features_contributing: List[str]

class ExoplanetAnomalyDetector:
    """Professional anomaly detection for exoplanet systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.pca = None
        self.anomaly_threshold = 0.1
        self.feature_names = [
            'period', 'duration', 'depth', 'planet_radius', 'stellar_radius',
            'stellar_temp', 'stellar_mag', 'impact_param', 'transit_snr',
            'num_transits', 'duty_cycle', 'log_period', 'log_planet_radius', 'log_depth'
        ]
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for anomaly detection
        
        Args:
            df: DataFrame with exoplanet data
            
        Returns:
            Numpy array of features
        """
        try:
            # Select and clean features
            available_features = [col for col in self.feature_names if col in df.columns]
            X = df[available_features].copy()
            
            # Handle missing values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Add derived features for anomaly detection
            X['period_depth_ratio'] = X['period'] / (X['depth'] + 1e-6)
            X['radius_temp_ratio'] = X['planet_radius'] / (X['stellar_temp'] / 1000)
            X['snr_duration_ratio'] = X['transit_snr'] / (X['duration'] + 1e-6)
            X['stellar_flux'] = (X['stellar_temp'] / 5778) ** 4 * (X['stellar_radius'] ** 2)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            self.logger.info(f"Prepared {X_scaled.shape[0]} samples for anomaly detection")
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return np.array([])
    
    def fit_isolation_forest(self, X: np.ndarray, contamination: float = 0.1) -> None:
        """
        Fit Isolation Forest for anomaly detection
        
        Args:
            X: Feature matrix
            contamination: Expected proportion of anomalies
        """
        try:
            self.isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            self.isolation_forest.fit(X)
            self.logger.info(f"Isolation Forest fitted with contamination={contamination}")
            
        except Exception as e:
            self.logger.error(f"Error fitting Isolation Forest: {e}")
    
    def detect_anomalies(self, X: np.ndarray) -> List[AnomalyResult]:
        """
        Detect anomalies in the dataset
        
        Args:
            X: Feature matrix
            
        Returns:
            List of AnomalyResult objects
        """
        try:
            if self.isolation_forest is None:
                raise ValueError("Isolation Forest not fitted. Call fit_isolation_forest first.")
            
            # Get anomaly scores and predictions
            anomaly_scores = self.isolation_forest.decision_function(X)
            anomaly_predictions = self.isolation_forest.predict(X)
            
            results = []
            for i, (score, prediction) in enumerate(zip(anomaly_scores, anomaly_predictions)):
                is_anomaly = prediction == -1
                confidence = abs(score)
                
                # Determine anomaly type based on features
                anomaly_type = self._classify_anomaly_type(X[i], score)
                
                # Get contributing features
                contributing_features = self._get_contributing_features(X[i])
                
                result = AnomalyResult(
                    is_anomaly=is_anomaly,
                    anomaly_score=score,
                    anomaly_type=anomaly_type,
                    confidence=confidence,
                    features_contributing=contributing_features
                )
                
                results.append(result)
            
            n_anomalies = sum(1 for r in results if r.is_anomaly)
            self.logger.info(f"Detected {n_anomalies} anomalies out of {len(results)} candidates")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _classify_anomaly_type(self, features: np.ndarray, score: float) -> str:
        """Classify the type of anomaly based on feature values"""
        try:
            # Map features to their names (simplified)
            feature_dict = dict(zip(self.feature_names, features[:len(self.feature_names)]))
            
            # Check for specific anomaly patterns
            if feature_dict.get('period', 0) > 1000:  # Very long period
                return "long_period_system"
            elif feature_dict.get('depth', 0) > 50000:  # Very deep transit
                return "deep_transit"
            elif feature_dict.get('planet_radius', 0) > 20:  # Very large planet
                return "giant_planet"
            elif feature_dict.get('transit_snr', 0) > 1000:  # Very high SNR
                return "high_snr_system"
            elif feature_dict.get('duty_cycle', 0) > 0.1:  # Very long transit
                return "long_transit"
            elif feature_dict.get('stellar_temp', 0) < 3000:  # Very cool star
                return "cool_star_system"
            elif feature_dict.get('stellar_temp', 0) > 8000:  # Very hot star
                return "hot_star_system"
            else:
                return "general_anomaly"
                
        except Exception as e:
            self.logger.error(f"Error classifying anomaly type: {e}")
            return "unknown_anomaly"
    
    def _get_contributing_features(self, features: np.ndarray) -> List[str]:
        """Get features that contribute most to the anomaly"""
        try:
            # Calculate feature importance (simplified)
            feature_importance = np.abs(features)
            
            # Get top contributing features
            top_indices = np.argsort(feature_importance)[-5:][::-1]
            
            contributing = []
            for idx in top_indices:
                if idx < len(self.feature_names):
                    contributing.append(self.feature_names[idx])
            
            return contributing
            
        except Exception as e:
            self.logger.error(f"Error getting contributing features: {e}")
            return []
    
    def detect_multi_planet_systems(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect potential multi-planet systems
        
        Args:
            df: DataFrame with exoplanet data
            
        Returns:
            DataFrame with multi-planet system flags
        """
        try:
            result_df = df.copy()
            result_df['is_multi_planet_system'] = False
            result_df['system_companions'] = 0
            
            # Group by stellar coordinates (simplified grouping)
            if 'stellar_temp' in df.columns and 'stellar_radius' in df.columns:
                # Group by stellar properties
                stellar_groups = df.groupby(['stellar_temp', 'stellar_radius'])
                
                for (temp, radius), group in stellar_groups:
                    if len(group) > 1:
                        # Multiple candidates around same star
                        result_df.loc[group.index, 'is_multi_planet_system'] = True
                        result_df.loc[group.index, 'system_companions'] = len(group) - 1
                        
                        self.logger.info(f"Found multi-planet system: {len(group)} candidates")
            
            n_multi_systems = result_df['is_multi_planet_system'].sum()
            self.logger.info(f"Detected {n_multi_systems} multi-planet systems")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error detecting multi-planet systems: {e}")
            return df
    
    def detect_rare_systems(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect rare and unusual exoplanet systems
        
        Args:
            df: DataFrame with exoplanet data
            
        Returns:
            DataFrame with rare system flags
        """
        try:
            result_df = df.copy()
            result_df['is_rare_system'] = False
            result_df['rare_system_type'] = 'normal'
            
            # Define rare system criteria
            rare_criteria = {
                'eccentric_orbit': (df.get('eccentricity', 0) > 0.5) if 'eccentricity' in df.columns else False,
                'ultra_short_period': df.get('period', 0) < 0.5,
                'ultra_long_period': df.get('period', 0) > 10000,
                'giant_planet': df.get('planet_radius', 0) > 10,
                'tiny_planet': df.get('planet_radius', 0) < 0.3,
                'hot_star': df.get('stellar_temp', 0) > 8000,
                'cool_star': df.get('stellar_temp', 0) < 3000,
                'deep_transit': df.get('depth', 0) > 100000,
                'shallow_transit': df.get('depth', 0) < 10
            }
            
            # Apply criteria
            for criterion, condition in rare_criteria.items():
                if isinstance(condition, pd.Series):
                    rare_mask = condition
                else:
                    rare_mask = pd.Series([False] * len(df))
                
                result_df.loc[rare_mask, 'is_rare_system'] = True
                result_df.loc[rare_mask, 'rare_system_type'] = criterion
            
            n_rare_systems = result_df['is_rare_system'].sum()
            self.logger.info(f"Detected {n_rare_systems} rare systems")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error detecting rare systems: {e}")
            return df
    
    def generate_anomaly_report(self, df: pd.DataFrame, anomaly_results: List[AnomalyResult]) -> str:
        """Generate comprehensive anomaly detection report"""
        try:
            n_anomalies = sum(1 for r in anomaly_results if r.is_anomaly)
            n_total = len(anomaly_results)
            
            report = f"""
🔍 EXOPLANET ANOMALY DETECTION REPORT
{'='*50}

📊 SUMMARY:
Total Candidates Analyzed: {n_total}
Anomalies Detected: {n_anomalies}
Anomaly Rate: {n_anomalies/n_total:.1%}

🚨 ANOMALY BREAKDOWN:
"""
            
            # Count anomaly types
            anomaly_types = {}
            for result in anomaly_results:
                if result.is_anomaly:
                    anomaly_type = result.anomaly_type
                    anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
            
            for anomaly_type, count in sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True):
                report += f"{anomaly_type}: {count} cases\n"
            
            # Top anomalies by score
            top_anomalies = sorted(
                [r for r in anomaly_results if r.is_anomaly],
                key=lambda x: x.anomaly_score
            )[:5]
            
            report += f"\n🔝 TOP ANOMALIES:\n"
            for i, anomaly in enumerate(top_anomalies, 1):
                report += f"{i}. Score: {anomaly.anomaly_score:.3f}, Type: {anomaly.anomaly_type}\n"
                report += f"   Contributing Features: {', '.join(anomaly.features_contributing[:3])}\n"
            
            # Multi-planet systems
            if 'is_multi_planet_system' in df.columns:
                n_multi = df['is_multi_planet_system'].sum()
                report += f"\n🌌 MULTI-PLANET SYSTEMS: {n_multi} detected\n"
            
            # Rare systems
            if 'is_rare_system' in df.columns:
                n_rare = df['is_rare_system'].sum()
                report += f"⭐ RARE SYSTEMS: {n_rare} detected\n"
            
            report += f"\n💡 INTERPRETATION:\n"
            if n_anomalies / n_total > 0.2:
                report += "⚠️  High anomaly rate detected. Review data quality and detection parameters.\n"
            elif n_anomalies / n_total > 0.1:
                report += "✅ Moderate anomaly rate. Some unusual systems detected.\n"
            else:
                report += "✅ Low anomaly rate. Most systems appear normal.\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating anomaly report: {e}")
            return f"Error generating report: {str(e)}"

def main():
    """Test the anomaly detection module"""
    detector = ExoplanetAnomalyDetector()
    
    print("🔍 Testing Exoplanet Anomaly Detector")
    print("="*40)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'period': np.random.lognormal(2, 1, n_samples),
        'duration': np.random.lognormal(1, 0.5, n_samples),
        'depth': np.random.lognormal(6, 1, n_samples),
        'planet_radius': np.random.lognormal(0, 0.5, n_samples),
        'stellar_radius': np.random.lognormal(0, 0.3, n_samples),
        'stellar_temp': np.random.normal(5800, 1000, n_samples),
        'stellar_mag': np.random.normal(12, 2, n_samples),
        'impact_param': np.random.uniform(0, 1, n_samples),
        'transit_snr': np.random.lognormal(3, 1, n_samples),
        'num_transits': np.random.poisson(20, n_samples),
        'duty_cycle': np.random.uniform(0.001, 0.1, n_samples),
        'log_period': np.random.normal(1, 0.5, n_samples),
        'log_planet_radius': np.random.normal(0, 0.3, n_samples),
        'log_depth': np.random.normal(3, 0.5, n_samples)
    })
    
    # Add some anomalies
    sample_data.loc[0, 'period'] = 0.1  # Ultra-short period
    sample_data.loc[1, 'planet_radius'] = 50  # Giant planet
    sample_data.loc[2, 'depth'] = 500000  # Very deep transit
    
    # Prepare features
    X = detector.prepare_features(sample_data)
    
    if len(X) > 0:
        # Fit isolation forest
        detector.fit_isolation_forest(X, contamination=0.05)
        
        # Detect anomalies
        anomalies = detector.detect_anomalies(X)
        
        # Generate report
        report = detector.generate_anomaly_report(sample_data, anomalies)
        print(report)
    else:
        print("❌ Error preparing features")

if __name__ == "__main__":
    main()
