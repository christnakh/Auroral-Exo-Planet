#!/usr/bin/env python3
"""
🔧 Model Accuracy Improvement Script
Addresses false positive issues in exoplanet classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedExoplanetTrainer:
    """Improved model trainer with better false positive handling"""
    
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
        
        # Check class distribution
        class_counts = self.df['label'].value_counts()
        logger.info(f"Class distribution: {dict(class_counts)}")
        
    def prepare_features(self):
        """Prepare features with improved preprocessing"""
        logger.info("Preparing features with improved preprocessing...")
        
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
        
        # Improved data cleaning
        # Replace infinite values with NaN first
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Use median imputation instead of zero filling
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                logger.info(f"Filled {X[col].isna().sum()} NaN values in {col} with median {median_val:.3f}")
        
        y = self.df['label']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Apply SMOTE on training set only to handle class imbalance
        try:
            smote = SMOTE(random_state=42)
            self.X_train_res, self.y_train_res = smote.fit_resample(self.X_train, self.y_train)
            logger.info(f"Resampled training set shape: {self.X_train_res.shape}; class distribution: {np.bincount(self.y_train_res)}")
        except Exception as e:
            logger.warning(f"SMOTE failed ({e}), falling back to original training set")
            self.X_train_res, self.y_train_res = self.X_train, self.y_train

        # Scale features (fit on resampled training only)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_res)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info(f"Training set shape: {self.X_train.shape}")
        logger.info(f"Test set shape: {self.X_test.shape}")
        logger.info(f"Class distribution in training set: {np.bincount(self.y_train)}")
        
    def train_improved_xgboost(self):
        """Train XGBoost with improved parameters for false positive reduction"""
        logger.info("Training improved XGBoost model...")
        
        # Calculate class weights to handle imbalance (use resampled labels)
        classes = np.unique(self.y_train_res)
        class_weights = compute_class_weight('balanced', classes=classes, y=self.y_train_res)
        weight_map = {cls: w for cls, w in zip(classes, class_weights)}

        # Base estimator
        xgb_base = xgb.XGBClassifier(
            n_estimators=200,  # More trees
            max_depth=8,        # Deeper trees
            learning_rate=0.05, # Lower learning rate
            subsample=0.8,      # Add subsampling
            colsample_bytree=0.8, # Feature subsampling
            reg_alpha=0.1,      # L1 regularization
            reg_lambda=1.0,     # L2 regularization
            n_jobs=-1,
            random_state=42,
            eval_metric='mlogloss'
        )

        # Train XGBoost with per-sample weights (multiclass handling)
        sample_weight = np.array([weight_map[y] for y in self.y_train_res], dtype=float)
        xgb_base.fit(self.X_train_scaled, self.y_train_res, sample_weight=sample_weight)
        self.models['xgboost_improved'] = xgb_base
        
        # Evaluate
        y_pred = xgb_base.predict(self.X_test_scaled)
        y_pred_proba = xgb_base.predict_proba(self.X_test_scaled)
        
        self.results['xgboost_improved'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': (y_pred == self.y_test).mean(),
            'auc': roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
        }
        
        logger.info(f"Improved XGBoost Accuracy: {self.results['xgboost_improved']['accuracy']:.4f}")
        logger.info(f"Improved XGBoost AUC: {self.results['xgboost_improved']['auc']:.4f}")
        
    def train_improved_random_forest(self):
        """Train Random Forest with improved parameters"""
        logger.info("Training improved Random Forest model...")
        
        # Base RF model with class_weight balanced
        rf_base = RandomForestClassifier(
            n_estimators=300,    # More trees
            max_depth=15,        # Deeper trees
            min_samples_split=5, # Prevent overfitting
            min_samples_leaf=2,   # Prevent overfitting
            max_features='sqrt', # Feature selection
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )

        # Calibrate RF probabilities
        rf_calibrated = CalibratedClassifierCV(estimator=rf_base, method='sigmoid', cv=3)
        rf_calibrated.fit(self.X_train_scaled, self.y_train_res)
        self.models['random_forest_improved'] = rf_calibrated
        
        # Evaluate
        y_pred = rf_calibrated.predict(self.X_test_scaled)
        y_pred_proba = rf_calibrated.predict_proba(self.X_test_scaled)
        
        self.results['random_forest_improved'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': (y_pred == self.y_test).mean(),
            'auc': roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
        }
        
        logger.info(f"Improved Random Forest Accuracy: {self.results['random_forest_improved']['accuracy']:.4f}")
        logger.info(f"Improved Random Forest AUC: {self.results['random_forest_improved']['auc']:.4f}")
        
    def train_ensemble_improved(self):
        """Train improved ensemble model"""
        logger.info("Training improved ensemble model...")
        
        # Get predictions from both models
        xgb_pred = self.models['xgboost_improved'].predict_proba(self.X_test)
        rf_pred = self.models['random_forest_improved'].predict_proba(self.X_test)
        
        # Weighted ensemble with better weights
        ensemble_pred = 0.6 * xgb_pred + 0.4 * rf_pred
        ensemble_pred_class = np.argmax(ensemble_pred, axis=1)
        
        self.results['ensemble_improved'] = {
            'predictions': ensemble_pred_class,
            'probabilities': ensemble_pred,
            'accuracy': (ensemble_pred_class == self.y_test).mean(),
            'auc': roc_auc_score(self.y_test, ensemble_pred, multi_class='ovr', average='weighted')
        }
        
        logger.info(f"Improved Ensemble Accuracy: {self.results['ensemble_improved']['accuracy']:.4f}")
        logger.info(f"Improved Ensemble AUC: {self.results['ensemble_improved']['auc']:.4f}")
        
    def plot_improved_confusion_matrices(self):
        """Plot improved confusion matrices"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            class_names = self.label_encoder.classes_
            
            models_to_plot = ['xgboost_improved', 'random_forest_improved', 'ensemble_improved']
            
            for i, model_name in enumerate(models_to_plot):
                if model_name in self.results:
                    cm = confusion_matrix(self.y_test, self.results[model_name]['predictions'])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=class_names, yticklabels=class_names, ax=axes[i])
                    axes[i].set_title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig('models/improved_confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Improved confusion matrices saved to models/improved_confusion_matrices.png")
        except Exception as e:
            logger.warning(f"Could not create improved confusion matrices: {e}")
        
    def generate_improved_report(self):
        """Generate detailed classification report for improved models"""
        logger.info("Generating improved classification reports...")
        
        for model_name, result in self.results.items():
            print(f"\n{model_name.upper()} IMPROVED CLASSIFICATION REPORT:")
            print("=" * 60)
            print(classification_report(
                self.y_test, 
                result['predictions'], 
                target_names=self.label_encoder.classes_
            ))
            
            # Calculate false positive rate specifically
            cm = confusion_matrix(self.y_test, result['predictions'])
            if cm.shape == (3, 3):  # 3x3 confusion matrix
                # False positive rate = FP / (FP + TN) for each class
                fp_rate_confirmed = cm[0, 2] / (cm[0, 2] + cm[0, 0] + cm[0, 1]) if (cm[0, 2] + cm[0, 0] + cm[0, 1]) > 0 else 0
                fp_rate_candidate = cm[1, 2] / (cm[1, 2] + cm[1, 0] + cm[1, 1]) if (cm[1, 2] + cm[1, 0] + cm[1, 1]) > 0 else 0
                fp_rate_false_pos = cm[2, 0] / (cm[2, 0] + cm[2, 1] + cm[2, 2]) if (cm[2, 0] + cm[2, 1] + cm[2, 2]) > 0 else 0
                
                print(f"False Positive Rates:")
                print(f"  Confirmed → False Positive: {fp_rate_confirmed:.3f}")
                print(f"  Candidate → False Positive: {fp_rate_candidate:.3f}")
                print(f"  False Positive → Confirmed: {fp_rate_false_pos:.3f}")
            
    def save_improved_models(self):
        """Save improved models"""
        logger.info("Saving improved models...")
        
        # Create models directory
        Path("models/trained_models").mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            joblib.dump(model, f"models/trained_models/{model_name}_model.pkl")
                
        # Save label encoder and scaler
        joblib.dump(self.label_encoder, "models/trained_models/improved_label_encoder.pkl")
        joblib.dump(self.scaler, "models/trained_models/improved_scaler.pkl")
        
        # Save results
        joblib.dump(self.results, "models/trained_models/improved_results.pkl")
        
        logger.info("Improved models saved successfully!")
        
    def train_all_improved_models(self):
        """Train all improved models and generate reports"""
        self.load_data()
        self.prepare_features()
        self.train_improved_xgboost()
        self.train_improved_random_forest()
        self.train_ensemble_improved()
        self.generate_improved_report()
        self.plot_improved_confusion_matrices()
        self.save_improved_models()

def main():
    """Main function to train improved models"""
    logger.info("🔧 Starting Improved Model Training...")
    trainer = ImprovedExoplanetTrainer()
    trainer.train_all_improved_models()
    logger.info("✅ Improved model training completed!")

if __name__ == "__main__":
    main()
