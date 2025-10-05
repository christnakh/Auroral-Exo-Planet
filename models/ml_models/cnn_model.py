"""
🔭 Light Curve CNN Model for Exoplanet Analysis
1D CNN for analyzing light curve time series data
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout, 
    Input, Concatenate, GlobalAveragePooling1D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightCurveCNN:
    """
    🔭 1D CNN for Light Curve Analysis
    
    This class implements a 1D Convolutional Neural Network specifically designed
    for analyzing exoplanet light curve time series data. It can detect transit
    signals, stellar variability, and other temporal patterns.
    """
    
    def __init__(self, sequence_length=1000, n_features=1, num_classes=3):
        """
        Initialize the CNN model
        
        Args:
            sequence_length (int): Length of input time series (default: 1000)
            n_features (int): Number of features per timestep (default: 1 for flux)
            num_classes (int): Number of output classes (default: 3)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.history = None
        
        logger.info(f"🔭 LightCurveCNN initialized: {sequence_length} timesteps, {n_features} features, {num_classes} classes")
    
    def _build_cnn_model(self):
        """
        Build the 1D CNN architecture
        
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        model = Sequential([
            # First Conv Block - Enhanced
            Conv1D(filters=64, kernel_size=7, activation='relu', 
                   input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=5, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            # Second Conv Block - Enhanced
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            BatchNormalization(),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            # Third Conv Block - Enhanced
            Conv1D(filters=256, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Fourth Conv Block
            Conv1D(filters=256, kernel_size=3, activation='relu'),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dropout(0.5),
            
            # Dense layers
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def preprocess_light_curves(self, light_curves, labels=None):
        """
        Preprocess light curve data for CNN input
        
        Args:
            light_curves (list): List of light curve arrays
            labels (array): Optional labels for supervised learning
            
        Returns:
            tuple: (X_processed, y_encoded) or (X_processed, None)
        """
        logger.info(f"🔧 Preprocessing {len(light_curves)} light curves...")
        
        processed_curves = []
        
        for i, curve in enumerate(light_curves):
            # Convert to numpy array if needed
            if not isinstance(curve, np.ndarray):
                curve = np.array(curve)
            
            # Normalize the light curve
            if len(curve) > 0:
                curve_normalized = (curve - np.mean(curve)) / (np.std(curve) + 1e-8)
            else:
                curve_normalized = np.zeros(self.sequence_length)
            
            # Pad or truncate to sequence_length
            if len(curve_normalized) > self.sequence_length:
                # Truncate
                curve_normalized = curve_normalized[:self.sequence_length]
            elif len(curve_normalized) < self.sequence_length:
                # Pad with zeros
                padding = self.sequence_length - len(curve_normalized)
                curve_normalized = np.pad(curve_normalized, (0, padding), mode='constant')
            
            processed_curves.append(curve_normalized)
        
        X_processed = np.array(processed_curves).reshape(-1, self.sequence_length, self.n_features)
        
        if labels is not None:
            # Encode labels
            y_encoded = tf.keras.utils.to_categorical(
                self.label_encoder.fit_transform(labels), 
                num_classes=self.num_classes
            )
            return X_processed, y_encoded
        else:
            return X_processed, None
    
    def train(self, light_curves, labels, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the CNN model
        
        Args:
            light_curves (list): List of light curve arrays
            labels (array): Ground truth labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data for validation
            
        Returns:
            dict: Training history
        """
        logger.info(f"🚀 Training CNN model for {epochs} epochs...")
        
        # Preprocess data
        X, y = self.preprocess_light_curves(light_curves, labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y.argmax(axis=1)
        )
        
        # Build model
        self.model = self._build_cnn_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ CNN training completed!")
        return self.history.history
    
    def evaluate(self, light_curves, labels):
        """
        Evaluate the trained model
        
        Args:
            light_curves (list): List of light curve arrays
            labels (array): Ground truth labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X, y = self.preprocess_light_curves(light_curves, labels)
        
        # Get predictions
        y_pred_proba = self.model.predict(X)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_true, y_pred)
        }
        
        logger.info(f"📊 CNN Evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return metrics
    
    def predict(self, light_curves):
        """
        Make predictions on new light curves
        
        Args:
            light_curves (list): List of light curve arrays
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X, _ = self.preprocess_light_curves(light_curves)
        
        # Get predictions
        probabilities = self.model.predict(X)
        predictions = np.argmax(probabilities, axis=1)
        
        # Convert back to class names
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities
    
    def save_model(self, path="models/trained_models/cnn_model.keras"):
        """
        Save the trained model and preprocessors
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save!")
        
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(path)
        
        # Save preprocessors
        joblib.dump(self.label_encoder, Path(path).parent / 'cnn_label_encoder.pkl')
        joblib.dump(self.scaler, Path(path).parent / 'cnn_scaler.pkl')
        
        logger.info(f"💾 CNN model saved to {path}")
    
    def load_model(self, path="models/trained_models/cnn_model.keras"):
        """
        Load a trained model and preprocessors
        
        Args:
            path (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(path)
        self.label_encoder = joblib.load(Path(path).parent / 'cnn_label_encoder.pkl')
        self.scaler = joblib.load(Path(path).parent / 'cnn_scaler.pkl')
        
        logger.info(f"📂 CNN model loaded from {path}")


class HybridModel:
    """
    🔭 Hybrid Model combining CNN and Tabular Features
    
    This class combines the CNN model for light curve analysis with
    traditional tabular features for comprehensive exoplanet classification.
    """
    
    def __init__(self, cnn_model, tabular_models, feature_names):
        """
        Initialize the hybrid model
        
        Args:
            cnn_model (LightCurveCNN): Trained CNN model
            tabular_models (dict): Dictionary of tabular models
            feature_names (list): List of tabular feature names
        """
        self.cnn_model = cnn_model
        self.tabular_models = tabular_models
        self.feature_names = feature_names
        self.class_names = ['confirmed', 'candidate', 'false_positive']
        
        logger.info("🔭 Hybrid model initialized with CNN and tabular models")
    
    def predict_hybrid(self, light_curves, tabular_features):
        """
        Make hybrid predictions combining CNN and tabular models
        
        Args:
            light_curves (list): List of light curve arrays
            tabular_features (array): Tabular feature matrix
            
        Returns:
            dict: Combined predictions and explanations
        """
        # CNN predictions
        cnn_predictions, cnn_probabilities = self.cnn_model.predict(light_curves)
        
        # Tabular predictions
        tabular_predictions = {}
        tabular_probabilities = {}
        
        for model_name, model in self.tabular_models.items():
            if model is not None:
                pred_proba = model.predict_proba(tabular_features)
                pred_labels = model.predict(tabular_features)
                
                tabular_predictions[model_name] = pred_labels
                tabular_probabilities[model_name] = pred_proba
        
        # Combine predictions (weighted average)
        combined_probabilities = np.zeros_like(cnn_probabilities)
        weights = {'cnn': 0.4, 'xgboost': 0.3, 'random_forest': 0.3}
        
        # Add CNN predictions
        combined_probabilities += weights['cnn'] * cnn_probabilities
        
        # Add tabular predictions
        for model_name, proba in tabular_probabilities.items():
            if model_name in weights:
                combined_probabilities += weights[model_name] * proba
        
        # Get final predictions
        final_predictions = np.argmax(combined_probabilities, axis=1)
        final_labels = [self.class_names[pred] for pred in final_predictions]
        
        return {
            'hybrid_predictions': final_labels,
            'hybrid_probabilities': combined_probabilities,
            'cnn_predictions': cnn_predictions,
            'cnn_probabilities': cnn_probabilities,
            'tabular_predictions': tabular_predictions,
            'tabular_probabilities': tabular_probabilities,
            'weights': weights
        }


def generate_synthetic_light_curve(length=1000, has_transit=True, noise_level=0.01, 
                                  transit_depth=0.01, transit_duration=0.1):
    """
    Generate synthetic light curve data for testing
    
    Args:
        length (int): Length of the light curve
        has_transit (bool): Whether to include a transit signal
        noise_level (float): Level of noise to add
        transit_depth (float): Depth of the transit
        transit_duration (float): Duration of the transit
        
    Returns:
        np.array: Synthetic light curve
    """
    # Base stellar variability
    time = np.linspace(0, 10, length)
    stellar_variability = 0.02 * np.sin(2 * np.pi * time) + 0.01 * np.sin(4 * np.pi * time)
    
    # Transit signal
    if has_transit:
        transit_center = length // 2
        transit_width = int(length * transit_duration)
        transit_start = max(0, transit_center - transit_width // 2)
        transit_end = min(length, transit_center + transit_width // 2)
        
        transit_signal = np.zeros(length)
        transit_signal[transit_start:transit_end] = -transit_depth
    else:
        transit_signal = np.zeros(length)
    
    # Combine signals
    light_curve = 1.0 + stellar_variability + transit_signal
    
    # Add noise
    noise = np.random.normal(0, noise_level, length)
    light_curve += noise
    
    return light_curve


def preprocess_light_curve(light_curve, target_length=1000):
    """
    Preprocess a single light curve for CNN input
    
    Args:
        light_curve (array): Input light curve
        target_length (int): Target length for padding/truncation
        
    Returns:
        np.array: Preprocessed light curve
    """
    # Normalize
    if len(light_curve) > 0:
        normalized = (light_curve - np.mean(light_curve)) / (np.std(light_curve) + 1e-8)
    else:
        normalized = np.zeros(target_length)
    
    # Pad or truncate
    if len(normalized) > target_length:
        normalized = normalized[:target_length]
    elif len(normalized) < target_length:
        padding = target_length - len(normalized)
        normalized = np.pad(normalized, (0, padding), mode='constant')
    
    return normalized


if __name__ == "__main__":
    """
    Demo script for CNN model training and evaluation
    """
    logger.info("🔭 Starting CNN Model Demo...")
    
    # Generate synthetic data
    logger.info("📊 Generating synthetic light curve data...")
    n_samples = 1000
    light_curves = []
    labels = []
    
    for i in range(n_samples):
        # Random parameters
        has_transit = np.random.choice([True, False], p=[0.6, 0.4])
        noise_level = np.random.uniform(0.005, 0.02)
        transit_depth = np.random.uniform(0.005, 0.05) if has_transit else 0
        
        # Generate light curve
        curve = generate_synthetic_light_curve(
            length=1000,
            has_transit=has_transit,
            noise_level=noise_level,
            transit_depth=transit_depth
        )
        light_curves.append(curve)
        
        # Create labels based on transit characteristics
        if has_transit and transit_depth > 0.02:
            labels.append('confirmed')
        elif has_transit:
            labels.append('candidate')
        else:
            labels.append('false_positive')
    
    # Train CNN model
    logger.info("🚀 Training CNN model...")
    cnn = LightCurveCNN(sequence_length=1000, n_features=1, num_classes=3)
    history = cnn.train(light_curves, labels, epochs=20, batch_size=32)
    
    # Evaluate model
    logger.info("📊 Evaluating CNN model...")
    metrics = cnn.evaluate(light_curves, labels)
    
    # Print results
    print("\n🔭 CNN Model Results:")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Save model
    cnn.save_model()
    logger.info("✅ CNN model training completed and saved!")