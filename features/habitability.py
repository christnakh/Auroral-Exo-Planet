"""
Planet Habitability Analysis Module
Calculates habitability scores and identifies potentially habitable exoplanets.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HabitabilityZone:
    """Habitability zone boundaries"""
    inner_edge: float  # AU
    outer_edge: float  # AU
    optimistic_inner: float  # AU (optimistic habitable zone)
    optimistic_outer: float  # AU (optimistic habitable zone)

class HabitabilityCalculator:
    """Professional habitability analysis for exoplanets"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Physical constants
        self.SOLAR_LUMINOSITY = 3.828e26  # W
        self.SOLAR_TEMPERATURE = 5778  # K
        self.STEFFAN_BOLTZMANN = 5.670374419e-8  # W⋅m⁻²⋅K⁻⁴
        self.AU = 1.496e11  # m
        
        # Habitability criteria
        self.MIN_HABITABLE_RADIUS = 0.5  # Earth radii
        self.MAX_HABITABLE_RADIUS = 2.5  # Earth radii
        self.MIN_STELLAR_TEMP = 2500  # K
        self.MAX_STELLAR_TEMP = 7200  # K
        
    def calculate_habitable_zone(self, stellar_temp: float, stellar_luminosity: float = None) -> HabitabilityZone:
        """
        Calculate habitable zone boundaries using Kopparapu et al. (2013) method
        
        Args:
            stellar_temp: Stellar effective temperature (K)
            stellar_luminosity: Stellar luminosity (L_sun), if None, estimated from temperature
            
        Returns:
            HabitabilityZone object with inner and outer edges
        """
        try:
            # Estimate luminosity from temperature if not provided
            if stellar_luminosity is None:
                # Using L ∝ T^4 relationship (simplified)
                stellar_luminosity = (stellar_temp / self.SOLAR_TEMPERATURE) ** 4
            
            # Kopparapu et al. (2013) habitable zone boundaries
            # Conservative habitable zone
            inner_edge = np.sqrt(stellar_luminosity / 1.1)  # AU
            outer_edge = np.sqrt(stellar_luminosity / 0.53)  # AU
            
            # Optimistic habitable zone
            optimistic_inner = np.sqrt(stellar_luminosity / 1.77)  # AU
            optimistic_outer = np.sqrt(stellar_luminosity / 0.32)  # AU
            
            return HabitabilityZone(
                inner_edge=inner_edge,
                outer_edge=outer_edge,
                optimistic_inner=optimistic_inner,
                optimistic_outer=optimistic_outer
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating habitable zone: {e}")
            return HabitabilityZone(0, 0, 0, 0)
    
    def calculate_equilibrium_temperature(self, stellar_temp: float, stellar_radius: float, 
                                        orbital_period: float, albedo: float = 0.3) -> float:
        """
        Calculate equilibrium temperature of the planet
        
        Args:
            stellar_temp: Stellar effective temperature (K)
            stellar_radius: Stellar radius (R_sun)
            orbital_period: Orbital period (days)
            albedo: Planetary albedo (default 0.3 for Earth-like)
            
        Returns:
            Equilibrium temperature (K)
        """
        try:
            # Convert orbital period to semi-major axis using Kepler's third law
            # a^3 = P^2 (in AU and years)
            period_years = orbital_period / 365.25
            semi_major_axis = (period_years ** 2) ** (1/3)  # AU
            
            # Calculate stellar flux
            stellar_flux = (stellar_temp / self.SOLAR_TEMPERATURE) ** 4 * (stellar_radius ** 2) / (semi_major_axis ** 2)
            
            # Equilibrium temperature
            teq = stellar_temp * ((1 - albedo) * stellar_flux) ** 0.25
            
            return teq
            
        except Exception as e:
            self.logger.error(f"Error calculating equilibrium temperature: {e}")
            return 0.0
    
    def calculate_habitability_score(self, planet_radius: float, stellar_temp: float, 
                                   stellar_radius: float, orbital_period: float,
                                   stellar_mag: float = None) -> Dict[str, float]:
        """
        Calculate comprehensive habitability score (0-1)
        
        Args:
            planet_radius: Planet radius (Earth radii)
            stellar_temp: Stellar temperature (K)
            stellar_radius: Stellar radius (R_sun)
            orbital_period: Orbital period (days)
            stellar_mag: Stellar magnitude (optional)
            
        Returns:
            Dictionary with habitability metrics
        """
        try:
            # Initialize scores
            scores = {
                'habitability_score': 0.0,
                'is_habitable': False,
                'habitable_zone_score': 0.0,
                'size_score': 0.0,
                'temperature_score': 0.0,
                'stellar_score': 0.0,
                'equilibrium_temp': 0.0,
                'habitable_zone_position': 'outside'
            }
            
            # 1. Size Score (0-1)
            if self.MIN_HABITABLE_RADIUS <= planet_radius <= self.MAX_HABITABLE_RADIUS:
                # Optimal range
                if 0.8 <= planet_radius <= 1.4:
                    size_score = 1.0
                else:
                    # Gradual decrease from optimal
                    size_score = 1.0 - abs(planet_radius - 1.1) / 0.3
            else:
                size_score = 0.0
            
            scores['size_score'] = max(0, min(1, size_score))
            
            # 2. Stellar Score (0-1)
            if self.MIN_STELLAR_TEMP <= stellar_temp <= self.MAX_STELLAR_TEMP:
                # Optimal stellar temperature (G-type stars)
                if 5000 <= stellar_temp <= 6500:
                    stellar_score = 1.0
                else:
                    # Gradual decrease from optimal
                    stellar_score = 1.0 - abs(stellar_temp - 5750) / 1000
            else:
                stellar_score = 0.0
            
            scores['stellar_score'] = max(0, min(1, stellar_score))
            
            # 3. Habitable Zone Score (0-1)
            hz = self.calculate_habitable_zone(stellar_temp)
            eq_temp = self.calculate_equilibrium_temperature(stellar_temp, stellar_radius, orbital_period)
            scores['equilibrium_temp'] = eq_temp
            
            # Check if planet is in habitable zone
            if hz.inner_edge > 0 and hz.outer_edge > 0:
                # Calculate orbital distance from period
                period_years = orbital_period / 365.25
                orbital_distance = (period_years ** 2) ** (1/3)  # AU
                
                if hz.inner_edge <= orbital_distance <= hz.outer_edge:
                    # Conservative habitable zone
                    hz_score = 1.0
                    scores['habitable_zone_position'] = 'conservative'
                elif hz.optimistic_inner <= orbital_distance <= hz.optimistic_outer:
                    # Optimistic habitable zone
                    hz_score = 0.7
                    scores['habitable_zone_position'] = 'optimistic'
                else:
                    hz_score = 0.0
                    if orbital_distance < hz.inner_edge:
                        scores['habitable_zone_position'] = 'too_close'
                    else:
                        scores['habitable_zone_position'] = 'too_far'
            else:
                hz_score = 0.0
                scores['habitable_zone_position'] = 'unknown'
            
            scores['habitable_zone_score'] = hz_score
            
            # 4. Temperature Score (0-1)
            if 200 <= eq_temp <= 400:  # K
                if 250 <= eq_temp <= 350:  # Optimal temperature range
                    temp_score = 1.0
                else:
                    # Gradual decrease from optimal
                    temp_score = 1.0 - abs(eq_temp - 300) / 50
            else:
                temp_score = 0.0
            
            scores['temperature_score'] = max(0, min(1, temp_score))
            
            # 5. Overall Habitability Score (weighted average)
            weights = {
                'size': 0.25,
                'stellar': 0.20,
                'habitable_zone': 0.35,
                'temperature': 0.20
            }
            
            overall_score = (
                scores['size_score'] * weights['size'] +
                scores['stellar_score'] * weights['stellar'] +
                scores['habitable_zone_score'] * weights['habitable_zone'] +
                scores['temperature_score'] * weights['temperature']
            )
            
            scores['habitability_score'] = overall_score
            scores['is_habitable'] = overall_score >= 0.6  # Threshold for habitability
            
            self.logger.info(f"Habitability analysis completed: score={overall_score:.3f}, habitable={scores['is_habitable']}")
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error calculating habitability score: {e}")
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
    
    def analyze_habitable_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze habitability for a batch of exoplanet candidates
        
        Args:
            df: DataFrame with exoplanet data
            
        Returns:
            DataFrame with habitability analysis added
        """
        try:
            self.logger.info(f"Analyzing habitability for {len(df)} candidates")
            
            results = []
            for idx, row in df.iterrows():
                try:
                    # Extract required parameters
                    planet_radius = row.get('planet_radius', 1.0)
                    stellar_temp = row.get('stellar_temp', 5800.0)
                    stellar_radius = row.get('stellar_radius', 1.0)
                    orbital_period = row.get('period', 365.0)
                    stellar_mag = row.get('stellar_mag', 12.0)
                    
                    # Calculate habitability
                    habitability = self.calculate_habitability_score(
                        planet_radius, stellar_temp, stellar_radius, orbital_period, stellar_mag
                    )
                    
                    # Add to results
                    result_row = row.to_dict()
                    result_row.update(habitability)
                    results.append(result_row)
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing candidate {idx}: {e}")
                    # Add default values
                    result_row = row.to_dict()
                    result_row.update({
                        'habitability_score': 0.0,
                        'is_habitable': False,
                        'habitable_zone_score': 0.0,
                        'size_score': 0.0,
                        'temperature_score': 0.0,
                        'stellar_score': 0.0,
                        'equilibrium_temp': 0.0,
                        'habitable_zone_position': 'error'
                    })
                    results.append(result_row)
            
            result_df = pd.DataFrame(results)
            self.logger.info(f"Habitability analysis completed: {result_df['is_habitable'].sum()} habitable candidates found")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in batch habitability analysis: {e}")
            return df

def main():
    """Test the habitability calculator"""
    calculator = HabitabilityCalculator()
    
    # Test case: Earth-like planet
    earth_like = calculator.calculate_habitability_score(
        planet_radius=1.0,
        stellar_temp=5778.0,
        stellar_radius=1.0,
        orbital_period=365.25
    )
    
    print("🌍 Earth-like Planet Analysis:")
    print(f"Habitability Score: {earth_like['habitability_score']:.3f}")
    print(f"Is Habitable: {earth_like['is_habitable']}")
    print(f"Equilibrium Temperature: {earth_like['equilibrium_temp']:.1f} K")
    print(f"HZ Position: {earth_like['habitable_zone_position']}")
    
    # Test case: Hot Jupiter
    hot_jupiter = calculator.calculate_habitability_score(
        planet_radius=10.0,
        stellar_temp=6000.0,
        stellar_radius=1.0,
        orbital_period=3.0
    )
    
    print("\n🔥 Hot Jupiter Analysis:")
    print(f"Habitability Score: {hot_jupiter['habitability_score']:.3f}")
    print(f"Is Habitable: {hot_jupiter['is_habitable']}")
    print(f"Equilibrium Temperature: {hot_jupiter['equilibrium_temp']:.1f} K")
    print(f"HZ Position: {hot_jupiter['habitable_zone_position']}")

if __name__ == "__main__":
    main()
