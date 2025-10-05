"""
Data preprocessing script for exoplanet ML model.
Handles KOI, TOI, and K2 datasets, standardizes labels, and creates features.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetDataProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.koi_df = None
        self.toi_df = None
        self.k2_df = None
        self.combined_df = None
        
    def load_datasets(self):
        """Load all three datasets"""
        logger.info("Loading datasets...")
        
        # Load KOI dataset
        koi_path = self.data_dir / "koi_cumulative.csv"
        if koi_path.exists():
            self.koi_df = pd.read_csv(koi_path)
            logger.info(f"Loaded KOI dataset: {self.koi_df.shape}")
        else:
            logger.error(f"KOI dataset not found at {koi_path}")
            
        # Load TOI dataset
        toi_path = self.data_dir / "toi.csv"
        if toi_path.exists():
            self.toi_df = pd.read_csv(toi_path)
            logger.info(f"Loaded TOI dataset: {self.toi_df.shape}")
        else:
            logger.error(f"TOI dataset not found at {toi_path}")
            
        # Load K2 dataset
        k2_path = self.data_dir / "k2_k2pandc.csv"
        if k2_path.exists():
            self.k2_df = pd.read_csv(k2_path)
            logger.info(f"Loaded K2 dataset: {self.k2_df.shape}")
        else:
            logger.error(f"K2 dataset not found at {k2_path}")
    
    def standardize_labels(self):
        """Standardize disposition labels across datasets"""
        logger.info("Standardizing labels...")
        
        # KOI dataset - use koi_disposition
        if self.koi_df is not None:
            self.koi_df['label'] = self.koi_df['koi_disposition'].str.lower()
            self.koi_df['mission'] = 'kepler'
            
        # TOI dataset - use tfopwg_disp
        if self.toi_df is not None:
            self.toi_df['label'] = self.toi_df['tfopwg_disp'].str.lower()
            self.toi_df['mission'] = 'tess'
            
        # K2 dataset - use disposition
        if self.k2_df is not None:
            self.k2_df['label'] = self.k2_df['disposition'].str.lower()
            self.k2_df['mission'] = 'k2'
    
    def create_features(self):
        """Create standardized features across all datasets"""
        logger.info("Creating features...")
        
        datasets = []
        
        # Process KOI dataset
        if self.koi_df is not None:
            koi_features = self._extract_koi_features()
            datasets.append(koi_features)
            
        # Process TOI dataset
        if self.toi_df is not None:
            toi_features = self._extract_toi_features()
            datasets.append(toi_features)
            
        # Process K2 dataset
        if self.k2_df is not None:
            k2_features = self._extract_k2_features()
            datasets.append(k2_features)
        
        # Combine all datasets
        if datasets:
            self.combined_df = pd.concat(datasets, ignore_index=True, sort=False)
            logger.info(f"Combined dataset shape: {self.combined_df.shape}")
        else:
            logger.error("No datasets to combine")
    
    def _extract_koi_features(self):
        """Extract features from KOI dataset"""
        df = self.koi_df.copy()
        
        # Create feature dictionary
        features = {
            'mission': df['mission'],
            'label': df['label'],
            'period': df['koi_period'],
            'duration': df['koi_duration'],
            'depth': df['koi_depth'],
            'planet_radius': df['koi_prad'],
            'stellar_radius': df['koi_srad'],
            'stellar_temp': df['koi_steff'],
            'stellar_mag': df['koi_kepmag'],
            'impact_param': df['koi_impact'],
            'transit_snr': df['koi_model_snr'],
            'num_transits': df['koi_num_transits']
        }
        
        # Create derived features
        features['duty_cycle'] = features['duration'] / features['period']
        features['log_period'] = np.log10(features['period'])
        features['log_planet_radius'] = np.log10(features['planet_radius'])
        features['log_depth'] = np.log10(features['depth'])
        
        return pd.DataFrame(features)
    
    def _extract_toi_features(self):
        """Extract features from TOI dataset"""
        df = self.toi_df.copy()
        
        # Create feature dictionary
        features = {
            'mission': df['mission'],
            'label': df['label'],
            'period': df['pl_orbper'],
            'duration': df['pl_trandurh'],
            'depth': df['pl_trandep'],
            'planet_radius': df['pl_rade'],
            'stellar_radius': df['st_rad'],
            'stellar_temp': df['st_teff'],
            'stellar_mag': df['st_tmag'],
            'impact_param': None,  # Not available in TOI
            'transit_snr': None,  # Not available in TOI
            'num_transits': None  # Not available in TOI
        }
        
        # Create derived features
        features['duty_cycle'] = features['duration'] / features['period']
        features['log_period'] = np.log10(features['period'])
        features['log_planet_radius'] = np.log10(features['planet_radius'])
        features['log_depth'] = np.log10(features['depth'])
        
        return pd.DataFrame(features)
    
    def _extract_k2_features(self):
        """Extract features from K2 dataset"""
        df = self.k2_df.copy()
        
        # Create feature dictionary
        features = {
            'mission': df['mission'],
            'label': df['label'],
            'period': df['pl_orbper'],
            'duration': df['pl_trandur'],
            'depth': df['pl_trandep'],
            'planet_radius': df['pl_rade'],
            'stellar_radius': df['st_rad'],
            'stellar_temp': df['st_teff'],
            'stellar_mag': df['sy_tmag'],
            'impact_param': df['pl_imppar'],
            'transit_snr': None,  # Not available in K2
            'num_transits': None  # Not available in K2
        }
        
        # Create derived features
        features['duty_cycle'] = features['duration'] / features['period']
        features['log_period'] = np.log10(features['period'])
        features['log_planet_radius'] = np.log10(features['planet_radius'])
        features['log_depth'] = np.log10(features['depth'])
        
        return pd.DataFrame(features)
    
    def clean_data(self):
        """Clean the combined dataset"""
        logger.info("Cleaning data...")
        
        if self.combined_df is None:
            logger.error("No combined dataset to clean")
            return
        
        # Remove rows with missing labels
        initial_shape = self.combined_df.shape
        self.combined_df = self.combined_df.dropna(subset=['label'])
        logger.info(f"Removed {initial_shape[0] - self.combined_df.shape[0]} rows with missing labels")
        
        # Standardize label values
        label_mapping = {
            'confirmed': 'confirmed',
            'candidate': 'candidate', 
            'false positive': 'false_positive',
            'fp': 'false_positive',
            'pc': 'candidate',
            'cp': 'candidate'
        }
        
        self.combined_df['label'] = self.combined_df['label'].map(label_mapping)
        
        # Remove rows with unmapped labels
        self.combined_df = self.combined_df.dropna(subset=['label'])
        
        # Handle missing values with median imputation
        numeric_columns = self.combined_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['mission']:  # Don't impute mission
                # Replace infinite values with NaN first
                self.combined_df[col] = self.combined_df[col].replace([np.inf, -np.inf], np.nan)
                # Then fill NaN with median
                self.combined_df[col] = self.combined_df[col].fillna(self.combined_df[col].median())
        
        logger.info(f"Final dataset shape: {self.combined_df.shape}")
        logger.info(f"Label distribution:\n{self.combined_df['label'].value_counts()}")
    
    def oversample_confirmed(self, target_ratio: float = 1.0, max_multiplier: int = 5):
        """Duplicate 'confirmed' rows to reach a desired ratio vs majority.
        
        Args:
            target_ratio: Desired confirmed/majority ratio after oversampling (<= 1 means up to majority).
            max_multiplier: Safety cap on number of times to duplicate to avoid runaway growth.
        """
        if self.combined_df is None or 'label' not in self.combined_df.columns:
            logger.warning("Cannot oversample: combined_df not ready or missing label column")
            return
        try:
            counts = self.combined_df['label'].value_counts()
            if 'confirmed' not in counts:
                logger.info("No confirmed class found; skipping oversample")
                return
            confirmed_count = int(counts.get('confirmed', 0))
            majority_count = int(counts.max())
            if confirmed_count <= 0:
                logger.info("Zero confirmed rows; skipping oversample")
                return
            target_count = min(int(majority_count * target_ratio), majority_count)
            if target_count <= confirmed_count:
                logger.info("Confirmed already meets target ratio; skipping oversample")
                return
            needed = target_count - confirmed_count
            confirmed_df = self.combined_df[self.combined_df['label'] == 'confirmed']
            multiplier = int(np.ceil(needed / confirmed_count))
            multiplier = int(max(1, min(multiplier, max_multiplier)))
            dup_df = pd.concat([confirmed_df] * multiplier, ignore_index=True).head(needed)
            before_shape = self.combined_df.shape
            self.combined_df = pd.concat([self.combined_df, dup_df], ignore_index=True)
            after_counts = self.combined_df['label'].value_counts()
            logger.info(
                f"Oversampled confirmed: +{len(dup_df)} rows; shape {before_shape} -> {self.combined_df.shape}; "
                f"counts {dict(counts)} -> {dict(after_counts)}"
            )
        except Exception as e:
            logger.warning(f"Oversample confirmed failed: {e}")
    
    def save_processed_data(self, output_path="data/processed_exoplanet_data.csv"):
        """Save the processed dataset"""
        if self.combined_df is not None:
            self.combined_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
        else:
            logger.error("No processed data to save")
    
    def get_data_summary(self):
        """Get summary statistics of the processed data"""
        if self.combined_df is None:
            return None
        
        summary = {
            'shape': self.combined_df.shape,
            'missing_values': self.combined_df.isnull().sum().sum(),
            'label_distribution': self.combined_df['label'].value_counts().to_dict(),
            'mission_distribution': self.combined_df['mission'].value_counts().to_dict(),
            'numeric_features': self.combined_df.select_dtypes(include=[np.number]).columns.tolist()
        }
        
        return summary

def main():
    """Main function to run data preprocessing"""
    processor = ExoplanetDataProcessor()
    
    # Load datasets
    processor.load_datasets()
    
    # Standardize labels
    processor.standardize_labels()
    
    # Create features
    processor.create_features()
    
    # Clean data
    processor.clean_data()
    
    # Optional: light oversampling of confirmed to balance dataset on disk
    try:
        processor.oversample_confirmed(target_ratio=1.0, max_multiplier=5)
    except Exception as e:
        logger.warning(f"Skipping oversample in CLI main: {e}")
    
    # Save processed data
    processor.save_processed_data()
    
    # Print summary
    summary = processor.get_data_summary()
    if summary:
        print("\nData Processing Summary:")
        print(f"Dataset shape: {summary['shape']}")
        print(f"Missing values: {summary['missing_values']}")
        print(f"Label distribution: {summary['label_distribution']}")
        print(f"Mission distribution: {summary['mission_distribution']}")

if __name__ == "__main__":
    main()
