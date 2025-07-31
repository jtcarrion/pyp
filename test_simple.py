#!/usr/bin/env python3
"""
Simplified 2D-Based Tomography Particle Tracking Test
=====================================================

This is a simplified version that avoids PYP dependencies and uses JSON instead.
"""

import os
import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import warnings
import time

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class SimpleParticleTracker2D:
    """
    Simplified 2D-based particle tracking for tomography
    """
    
    def __init__(self, parameters: Dict):
        """
        Initialize the particle tracker
        
        Parameters
        ----------
        parameters : Dict
            Parameters dictionary
        """
        self.parameters = parameters
        self.validation_results = {}
        
    def validate_allbox_format(self, allboxes: np.ndarray) -> bool:
        """
        Advanced validation of allbox format
        
        Parameters
        ----------
        allboxes : np.ndarray
            Array of particle coordinates
            
        Returns
        -------
        bool
            True if valid, raises ValidationError if not
        """
        try:
            # Check basic shape
            if allboxes.ndim != 2:
                raise ValidationError(f"Expected 2D array, got {allboxes.ndim}D")
            
            if allboxes.shape[1] != 4:
                raise ValidationError(f"Expected 4 columns (X, Y, frame, tilt), got {allboxes.shape[1]}")
            
            # Check for NaN values
            if np.any(np.isnan(allboxes)):
                raise ValidationError("Found NaN values in allbox data")
            
            # Check coordinate ranges (reasonable bounds)
            x_coords = allboxes[:, 0]
            y_coords = allboxes[:, 1]
            
            if np.any(x_coords < 0) or np.any(y_coords < 0):
                logger.warning("Found negative coordinates - this might be intentional")
            
            # Check tilt angle ranges
            tilt_angles = allboxes[:, 3]
            if np.any(np.abs(tilt_angles) > 90):
                raise ValidationError("Tilt angles should be between -90 and 90 degrees")
            
            return True
            
        except Exception as e:
            raise ValidationError(f"Validation failed: {e}")
    
    def read_allbox_file(self, allbox_file: str) -> np.ndarray:
        """
        Task 1: Read allbox file with particle coordinates
        
        Expected format: [X-coord, Y-coord, frame=0, tilt-angle]
        
        Parameters
        ----------
        allbox_file : str
            Path to allbox file
            
        Returns
        -------
        np.ndarray
            Array of particle coordinates
        """
        logger.info(f"Reading allbox file: {allbox_file}")
        
        try:
            # Check if file exists
            if not os.path.exists(allbox_file):
                raise FileNotFoundError(f"Allbox file not found: {allbox_file}")
            
            # Read allbox file
            allboxes = np.loadtxt(allbox_file, ndmin=2)
            
            # Validate format
            self.validate_allbox_format(allboxes)
            
            # Set frame to 0 for all particles (average across frames)
            allboxes[:, 2] = 0
            
            # Store validation results
            self.validation_results['allbox'] = {
                'num_particles': len(allboxes),
                'x_range': (np.min(allboxes[:, 0]), np.max(allboxes[:, 0])),
                'y_range': (np.min(allboxes[:, 1]), np.max(allboxes[:, 1])),
                'tilt_range': (np.min(allboxes[:, 3]), np.max(allboxes[:, 3])),
                'unique_tilts': len(np.unique(allboxes[:, 3]))
            }
            
            logger.info(f"Loaded {len(allboxes)} particles from allbox file")
            logger.info(f"Coordinate ranges: X={self.validation_results['allbox']['x_range']}, "
                       f"Y={self.validation_results['allbox']['y_range']}")
            logger.info(f"Tilt range: {self.validation_results['allbox']['tilt_range']}")
            
            return allboxes
            
        except Exception as e:
            logger.error(f"Error reading allbox file: {e}")
            raise
    
    def validate_tracked_particles(self, tracked_particles: Dict) -> bool:
        """
        Advanced validation of tracked particles format
        
        Parameters
        ----------
        tracked_particles : Dict
            Dictionary of tracked particles
            
        Returns
        -------
        bool
            True if valid, raises ValidationError if not
        """
        try:
            if not isinstance(tracked_particles, dict):
                raise ValidationError("Tracked particles must be a dictionary")
            
            if len(tracked_particles) == 0:
                raise ValidationError("No tracked particles found")
            
            all_tilt_angles = set()
            total_positions = 0
            
            for particle_id, tilt_data in tracked_particles.items():
                if not isinstance(tilt_data, dict):
                    raise ValidationError(f"Invalid format for particle {particle_id}")
                
                for tilt_key, position in tilt_data.items():
                    # Validate tilt key format
                    if not tilt_key.startswith('tilt_'):
                        raise ValidationError(f"Invalid tilt key format: {tilt_key}")
                    
                    try:
                        tilt_angle = float(tilt_key.replace('tilt_', ''))
                        all_tilt_angles.add(tilt_angle)
                    except ValueError:
                        raise ValidationError(f"Could not parse tilt angle from key: {tilt_key}")
                    
                    # Validate position format
                    if not isinstance(position, (tuple, list)) or len(position) != 2:
                        raise ValidationError(f"Invalid position format for {tilt_key}: {position}")
                    
                    x, y = position
                    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                        raise ValidationError(f"Invalid coordinate types for {tilt_key}: {position}")
                    
                    total_positions += 1
            
            # Store validation results
            self.validation_results['tracked_particles'] = {
                'num_particles': len(tracked_particles),
                'num_positions': total_positions,
                'unique_tilts': len(all_tilt_angles),
                'tilt_range': (min(all_tilt_angles), max(all_tilt_angles))
            }
            
            return True
            
        except Exception as e:
            raise ValidationError(f"Validation failed: {e}")
    
    def read_tracked_particles_pkl(self, pkl_file: str) -> Dict:
        """
        Task 2: Read tracked particles from PKL file
        
        Expected format: 
        {
            "particle_1": {
                "tilt_-40": (x1, y1),
                "tilt_-38": (x2, y2),
                ...
            },
            "particle_2": {
                "tilt_-40": (x3, y3),
                "tilt_-38": (x4, y4),
                ...
            }
        }
        
        Parameters
        ----------
        pkl_file : str
            Path to PKL file with tracked particles
            
        Returns
        -------
        Dict
            Dictionary of tracked particles with PTLIND assignments
        """
        logger.info(f"Reading tracked particles from PKL file: {pkl_file}")
        
        try:
            # Check if file exists
            if not os.path.exists(pkl_file):
                raise FileNotFoundError(f"PKL file not found: {pkl_file}")
            
            # Read PKL file
            with open(pkl_file, 'rb') as f:
                tracked_particles = pickle.load(f)
            
            # Validate format
            self.validate_tracked_particles(tracked_particles)
            
            # Process tracked particles with progress bar
            processed_particles = {}
            ptlind_counter = 1
            
            with tqdm(total=len(tracked_particles), desc="Processing particles") as pbar:
                for particle_id, tilt_data in tracked_particles.items():
                    processed_particles[particle_id] = {
                        'ptlind': ptlind_counter,
                        'tilts': {}
                    }
                    
                    for tilt_key, (x, y) in tilt_data.items():
                        # Extract tilt angle from key (e.g., "tilt_-40" -> -40)
                        try:
                            tilt_angle = float(tilt_key.replace('tilt_', ''))
                        except ValueError:
                            logger.warning(f"Could not parse tilt angle from key: {tilt_key}")
                            continue
                        
                        processed_particles[particle_id]['tilts'][tilt_angle] = (x, y)
                    
                    ptlind_counter += 1
                    pbar.update(1)
            
            logger.info(f"Processed {len(processed_particles)} tracked particles")
            logger.info(f"Total positions: {self.validation_results['tracked_particles']['num_positions']}")
            logger.info(f"Unique tilts: {self.validation_results['tracked_particles']['unique_tilts']}")
            
            return processed_particles
            
        except Exception as e:
            logger.error(f"Error reading PKL file: {e}")
            raise
    
    def convert_to_json_format(self, allboxes: np.ndarray, tracked_particles: Dict) -> Dict:
        """
        Task 3: Convert to JSON format with PTLIND constraints
        
        Parameters
        ----------
        allboxes : np.ndarray
            Array of particle coordinates from allbox file
        tracked_particles : Dict
            Dictionary of tracked particles with PTLIND assignments
            
        Returns
        -------
        Dict
            JSON-compatible dictionary for PYP format
        """
        logger.info("Converting to JSON format")
        
        particle_data = []
        tilt_data = {}
        particle_counter = 1
        
        # Process tracked particles with progress bar
        total_particles = sum(len(particle_info['tilts']) for particle_info in tracked_particles.values())
        
        with tqdm(total=total_particles, desc="Creating particle data") as pbar:
            for particle_id, particle_info in tracked_particles.items():
                ptlind = particle_info['ptlind']
                
                for tilt_angle, (x, y) in particle_info['tilts'].items():
                    # Create particle object
                    particle = {
                        "particle_index": particle_counter,
                        "shift_x": 0.0,
                        "shift_y": 0.0,
                        "shift_z": 0.0,
                        "psi": 0.0,
                        "theta": 0.0,
                        "phi": 0.0,
                        "x_position_3d": float(x),
                        "y_position_3d": float(y),
                        "z_position_3d": 0.0,  # 2D-based tomography
                        "score": 0.0,
                        "occ": 100.0,
                        "ptlind": ptlind
                    }
                    
                    particle_data.append(particle)
                    
                    # Create tilt parameter
                    if tilt_angle not in tilt_data:
                        tilt_data[tilt_angle] = {}
                    
                    region_index = 0  # Default region
                    tilt_data[tilt_angle][region_index] = {
                        "tilt_index": tilt_angle,
                        "region_index": region_index,
                        "shift_x": 0.0,
                        "shift_y": 0.0,
                        "angle": tilt_angle,
                        "axis": 0.0
                    }
                    
                    particle_counter += 1
                    pbar.update(1)
        
        result = {
            "particles": particle_data,
            "tilts": tilt_data,
            "metadata": {
                "total_particles": len(particle_data),
                "total_tilts": len(tilt_data),
                "validation_results": self.validation_results
            }
        }
        
        logger.info(f"Created {len(particle_data)} particle entries")
        logger.info(f"Created {len(tilt_data)} tilt entries")
        
        return result
    
    def create_parameter_file(self, output_file: str, allboxes: np.ndarray, tracked_particles: Dict):
        """
        Create JSON parameter file
        
        Parameters
        ----------
        output_file : str
            Output parameter file path
        allboxes : np.ndarray
            Array of particle coordinates
        tracked_particles : Dict
            Dictionary of tracked particles
        """
        logger.info(f"Creating parameter file: {output_file}")
        
        start_time = time.time()
        
        # Convert to JSON format
        json_data = self.convert_to_json_format(allboxes, tracked_particles)
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Parameter file created: {output_file} (took {elapsed_time:.2f}s)")
    
    def generate_quality_report(self) -> Dict:
        """
        Generate quality assessment report
        
        Returns
        -------
        Dict
            Quality assessment report
        """
        report = {
            'validation_results': self.validation_results,
            'summary': {
                'total_particles': len(self.validation_results.get('tracked_particles', {}).get('num_particles', 0)),
                'total_tilts': len(self.validation_results.get('tracked_particles', {}).get('unique_tilts', 0)),
                'coverage': self._calculate_coverage(),
                'consistency': self._check_consistency()
            }
        }
        
        return report
    
    def _calculate_coverage(self) -> float:
        """Calculate particle coverage across tilts"""
        if 'tracked_particles' not in self.validation_results:
            return 0.0
        
        total_positions = self.validation_results['tracked_particles']['num_positions']
        num_particles = self.validation_results['tracked_particles']['num_particles']
        num_tilts = self.validation_results['tracked_particles']['unique_tilts']
        
        expected_positions = num_particles * num_tilts
        coverage = total_positions / expected_positions if expected_positions > 0 else 0.0
        
        return coverage
    
    def _check_consistency(self) -> Dict:
        """Check data consistency"""
        consistency = {
            'allbox_tilt_range': self.validation_results.get('allbox', {}).get('tilt_range', (0, 0)),
            'tracked_tilt_range': self.validation_results.get('tracked_particles', {}).get('tilt_range', (0, 0)),
            'tilt_overlap': True  # Will be calculated
        }
        
        # Check tilt range overlap
        allbox_min, allbox_max = consistency['allbox_tilt_range']
        tracked_min, tracked_max = consistency['tracked_tilt_range']
        
        overlap_min = max(allbox_min, tracked_min)
        overlap_max = min(allbox_max, tracked_max)
        
        consistency['tilt_overlap'] = overlap_max >= overlap_min
        consistency['overlap_range'] = (overlap_min, overlap_max)
        
        return consistency


def main():
    """
    Main function to run the simplified workflow
    """
    # Example usage
    allbox_file = "particles.allbox"
    pkl_file = "tracked_particles.pkl"
    output_file = "test_dataset_r01_02.json"
    
    # Load configuration if available
    config_file = "test_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            parameters = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
    else:
        # Default parameters
        parameters = {
            "data_set": "test_dataset",
            "slurm_tasks": 4,
            "slurm_verbose": True,
        }
        logger.info("Using default configuration")
    
    # Initialize particle tracker
    tracker = SimpleParticleTracker2D(parameters)
    
    try:
        # Task 1: Read allbox file
        logger.info("=== Task 1: Reading allbox file ===")
        allboxes = tracker.read_allbox_file(allbox_file)
        
        # Task 2: Read tracked particles
        logger.info("=== Task 2: Reading tracked particles ===")
        tracked_particles = tracker.read_tracked_particles_pkl(pkl_file)
        
        # Task 3: Convert to JSON format
        logger.info("=== Task 3: Converting to JSON format ===")
        tracker.create_parameter_file(output_file, allboxes, tracked_particles)
        
        # Generate quality report
        logger.info("=== Generating Quality Report ===")
        quality_report = tracker.generate_quality_report()
        logger.info(f"Quality Report: {json.dumps(quality_report, indent=2)}")
        
        logger.info("All tasks completed successfully!")
        logger.info(f"Output file: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main workflow: {e}")
        raise


if __name__ == "__main__":
    main() 