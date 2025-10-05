"""
Model training script for exoplanet classification.
Trains XGBoost, Random Forest, and ensemble models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetModelTrainer:
    def __init__(self, data_path="data/processed_exoplanet_data.csv"):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the processed dataset"""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset with shape: {self.df.shape}")
        
    def prepare_features(self):
        """Prepare features for training"""
        logger.info("Preparing features...")
        
        # Select numeric features for training
        feature_columns = [
            'period', 'duration', 'depth', 'planet_radius', 'stellar_radius',
            'stellar_temp', 'stellar_mag', 'impact_param', 'transit_snr',
            'num_transits', 'duty_cycle', 'log_period', 'log_planet_radius', 'log_depth'
        ]
        
        # Filter out columns that don't exist
        available_features = [col for col in feature_columns if col in self.df.columns]
        logger.info(f"Using features: {available_features}")
        
        # Create feature matrix
        X = self.df[available_features].copy()
        # Replace infinite values with NaN first
        X = X.replace([np.inf, -np.inf], np.nan)
        # Fill any remaining NaN with 0
        X = X.fillna(0)
        y = self.df['label']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Optional: Oversample 'confirmed' class in the training set to reduce imbalance
        try:
            class_names = list(self.label_encoder.classes_)
            if 'confirmed' in class_names:
                confirmed_idx = int(np.where(self.label_encoder.classes_ == 'confirmed')[0][0])
                train_counts = np.bincount(self.y_train)
                majority_count = int(train_counts.max())
                if confirmed_idx < len(train_counts) and train_counts[confirmed_idx] < majority_count:
                    ros = RandomOverSampler(sampling_strategy={confirmed_idx: majority_count}, random_state=42)
                    self.X_train, self.y_train = ros.fit_resample(self.X_train, self.y_train)
                    logger.info(f"Applied oversampling: confirmed from {train_counts[confirmed_idx]} to {majority_count}")
        except Exception as e:
            logger.warning(f"Oversampling skipped due to error: {e}")

        # Scale features (fit on potentially oversampled training set)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info(f"Training set shape: {self.X_train.shape}")
        logger.info(f"Test set shape: {self.X_test.shape}")
        logger.info(f"Class distribution in training set: {np.bincount(self.y_train)}")
        
    def train_xgboost(self):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        xgb_model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = xgb_model
        
        # Evaluate
        y_pred = xgb_model.predict(self.X_test)
        y_pred_proba = xgb_model.predict_proba(self.X_test)
        
        self.results['xgboost'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': (y_pred == self.y_test).mean(),
            'auc': roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
        }
        
        logger.info(f"XGBoost Accuracy: {self.results['xgboost']['accuracy']:.4f}")
        logger.info(f"XGBoost AUC: {self.results['xgboost']['auc']:.4f}")
        
    def train_random_forest(self):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        rf_model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf_model
        
        # Evaluate
        y_pred = rf_model.predict(self.X_test)
        y_pred_proba = rf_model.predict_proba(self.X_test)
        
        self.results['random_forest'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': (y_pred == self.y_test).mean(),
            'auc': roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
        }
        
        logger.info(f"Random Forest Accuracy: {self.results['random_forest']['accuracy']:.4f}")
        logger.info(f"Random Forest AUC: {self.results['random_forest']['auc']:.4f}")
        
    def train_ensemble(self):
        """Train ensemble model"""
        logger.info("Training ensemble model...")
        
        # Get predictions from both models
        xgb_pred = self.models['xgboost'].predict_proba(self.X_test)
        rf_pred = self.models['random_forest'].predict_proba(self.X_test)
        
        # Simple average ensemble
        ensemble_pred = (xgb_pred + rf_pred) / 2
        ensemble_pred_class = np.argmax(ensemble_pred, axis=1)
        
        self.results['ensemble'] = {
            'predictions': ensemble_pred_class,
            'probabilities': ensemble_pred,
            'accuracy': (ensemble_pred_class == self.y_test).mean(),
            'auc': roc_auc_score(self.y_test, ensemble_pred, multi_class='ovr', average='weighted')
        }
        
        logger.info(f"Ensemble Accuracy: {self.results['ensemble']['accuracy']:.4f}")
        logger.info(f"Ensemble AUC: {self.results['ensemble']['auc']:.4f}")
        
    def cross_validate(self):
        """Perform cross-validation"""
        logger.info("Performing cross-validation...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in self.models.items():
            if model_name == 'ensemble':
                continue
                
            try:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
                logger.info(f"{model_name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            except Exception as e:
                logger.warning(f"Cross-validation failed for {model_name}: {e}")
            
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            class_names = self.label_encoder.classes_
            
            for i, (model_name, result) in enumerate(self.results.items()):
                cm = confusion_matrix(self.y_test, result['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names, yticklabels=class_names, ax=axes[i])
                axes[i].set_title(f'{model_name.title()} Confusion Matrix')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to avoid display issues
            logger.info("Confusion matrices saved to models/confusion_matrices.png")
        except Exception as e:
            logger.warning(f"Could not create confusion matrices: {e}")
        
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # XGBoost feature importance
            xgb_importance = self.models['xgboost'].feature_importances_
            feature_names = self.X_train.columns
            xgb_df = pd.DataFrame({'feature': feature_names, 'importance': xgb_importance})
            xgb_df = xgb_df.sort_values('importance', ascending=True)
            
            axes[0].barh(xgb_df['feature'], xgb_df['importance'])
            axes[0].set_title('XGBoost Feature Importance')
            axes[0].set_xlabel('Importance')
            
            # Random Forest feature importance
            rf_importance = self.models['random_forest'].feature_importances_
            rf_df = pd.DataFrame({'feature': feature_names, 'importance': rf_importance})
            rf_df = rf_df.sort_values('importance', ascending=True)
            
            axes[1].barh(rf_df['feature'], rf_df['importance'])
            axes[1].set_title('Random Forest Feature Importance')
            axes[1].set_xlabel('Importance')
            
            plt.tight_layout()
            plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to avoid display issues
            logger.info("Feature importance plots saved to models/feature_importance.png")
        except Exception as e:
            logger.warning(f"Could not create feature importance plots: {e}")
        
    def generate_report(self):
        """Generate detailed classification report"""
        logger.info("Generating classification reports...")
        
        for model_name, result in self.results.items():
            print(f"\n{model_name.upper()} CLASSIFICATION REPORT:")
            print("=" * 50)
            print(classification_report(
                self.y_test, 
                result['predictions'], 
                target_names=self.label_encoder.classes_
            ))
            
    def save_models(self):
        """Save trained models"""
        logger.info("Saving models...")
        
        # Create models directory
        Path("models/trained_models").mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            if model_name != 'ensemble':
                joblib.dump(model, f"models/trained_models/{model_name}_model.pkl")
                
        # Save label encoder and scaler
        joblib.dump(self.label_encoder, "models/trained_models/label_encoder.pkl")
        joblib.dump(self.scaler, "models/trained_models/scaler.pkl")
        
        # Save results
        joblib.dump(self.results, "models/trained_models/results.pkl")
        
        logger.info("Models saved successfully!")
        
    def train_all_models(self):
        """Train all models and generate reports"""
        self.load_data()
        self.prepare_features()
        self.train_xgboost()
        self.train_random_forest()
        self.train_ensemble()
        self.cross_validate()
        self.generate_report()
        self.plot_confusion_matrices()
        self.plot_feature_importance()
        self.save_models()

def main():
    """Main function to train all models"""
    trainer = ExoplanetModelTrainer()
    trainer.train_all_models()

if __name__ == "__main__":
    main()
