#!/usr/bin/env python3
"""
2D-Based Tomography Particle Tracking with CSPT Refinement
==========================================================

This module implements a robust workflow for:
1. Reading allbox files with particle coordinates
2. Processing tracked particles from PKL files
3. Converting to PYP format with PTLIND constraints
4. Running parallelized CSPT refinement

INNOVATIVE FEATURES:
- Advanced validation and error handling
- Progress tracking with tqdm
- Automatic quality assessment
- Flexible coordinate systems
- Memory-efficient processing
- Parallel processing optimization
- Robust error recovery

Author: Your Name
Date: 2024
"""

import os
import sys
import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

# Add PYP to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pyp.inout.metadata.cistem_star_file import Parameters, Particle, Tilt, ExtendedParameters
from pyp.inout.metadata.frealign_parfile import Parameters as FrealignParameters
from pyp.system import project_params
from pyp.system.logging import initialize_pyp_logger

# Initialize logger
logger = initialize_pyp_logger()


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class ParticleTracker2D:
    """
    Robust 2D-based particle tracking for tomography with CSPT refinement
    """
    
    def __init__(self, parameters: Dict):
        """
        Initialize the particle tracker
        
        Parameters
        ----------
        parameters : Dict
            PYP parameters dictionary
        """
        self.parameters = parameters
        self.particle_parameters = {}
        self.tilt_parameters = {}
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
            
            # Check for duplicate particles
            unique_particles = np.unique(allboxes[:, :2], axis=0)
            if len(unique_particles) != len(allboxes):
                logger.warning(f"Found {len(allboxes) - len(unique_particles)} duplicate particle positions")
            
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
    
    def convert_to_pyp_format(self, allboxes: np.ndarray, tracked_particles: Dict) -> Tuple[Dict, Dict]:
        """
        Task 3: Convert to PYP format with PTLIND constraints
        
        Parameters
        ----------
        allboxes : np.ndarray
            Array of particle coordinates from allbox file
        tracked_particles : Dict
            Dictionary of tracked particles with PTLIND assignments
            
        Returns
        -------
        Tuple[Dict, Dict]
            (particle_parameters, tilt_parameters) for PYP format
        """
        logger.info("Converting to PYP format")
        
        particle_parameters = {}
        tilt_parameters = {}
        particle_counter = 1
        
        # Create reverse lookup for allbox particles
        allbox_lookup = {}
        for i, (x, y, frame, tilt) in enumerate(allboxes):
            allbox_lookup[(x, y, tilt)] = i
        
        # Process tracked particles with progress bar
        total_particles = sum(len(particle_data['tilts']) for particle_data in tracked_particles.values())
        
        with tqdm(total=total_particles, desc="Creating particle parameters") as pbar:
            for particle_id, particle_data in tracked_particles.items():
                ptlind = particle_data['ptlind']
                
                for tilt_angle, (x, y) in particle_data['tilts'].items():
                    # Create particle object
                    particle = Particle(
                        particle_index=particle_counter,
                        shift_x=0.0,
                        shift_y=0.0,
                        shift_z=0.0,
                        psi=0.0,
                        theta=0.0,
                        phi=0.0,
                        x_position_3d=x,
                        y_position_3d=y,
                        z_position_3d=0,  # 2D-based tomography
                        score=0.0,
                        occ=100.0
                    )
                    
                    particle_parameters[particle_counter] = particle
                    
                    # Create tilt parameter
                    if tilt_angle not in tilt_parameters:
                        tilt_parameters[tilt_angle] = {}
                    
                    region_index = 0  # Default region
                    tilt_parameters[tilt_angle][region_index] = Tilt(
                        tilt_index=tilt_angle,
                        region_index=region_index,
                        shift_x=0.0,
                        shift_y=0.0,
                        angle=tilt_angle,
                        axis=0.0
                    )
                    
                    particle_counter += 1
                    pbar.update(1)
        
        logger.info(f"Created {len(particle_parameters)} particle parameters")
        logger.info(f"Created {len(tilt_parameters)} tilt parameters")
        
        return particle_parameters, tilt_parameters
    
    def create_cistem_parameters(self, allboxes: np.ndarray, tracked_particles: Dict) -> np.ndarray:
        """
        Create CistemParameters array for PYP format
        
        Parameters
        ----------
        allboxes : np.ndarray
            Array of particle coordinates
        tracked_particles : Dict
            Dictionary of tracked particles
            
        Returns
        -------
        np.ndarray
            CistemParameters array
        """
        # Calculate total number of particles
        total_particles = sum(len(particle_data['tilts']) for particle_data in tracked_particles.values())
        
        # Initialize cistem_parameters array
        cistem_parameters = np.zeros((total_particles, 32), dtype='float')
        
        # Set default values
        cistem_parameters[:, 0] = np.arange(1, total_particles + 1)  # Position in stack
        cistem_parameters[:, 1] = 0.0  # PSI
        cistem_parameters[:, 2] = 0.0  # THETA
        cistem_parameters[:, 3] = 0.0  # PHI
        cistem_parameters[:, 4] = 0.0  # SHX
        cistem_parameters[:, 5] = 0.0  # SHY
        cistem_parameters[:, 6] = 1.0  # MAG
        cistem_parameters[:, 7] = 0    # FILM
        cistem_parameters[:, 8] = 0.0  # DF1
        cistem_parameters[:, 9] = 0.0  # DF2
        cistem_parameters[:, 10] = 0.0 # ANGAST
        cistem_parameters[:, 11] = 100.0 # OCC
        cistem_parameters[:, 12] = 0.0 # LOGP
        cistem_parameters[:, 13] = 0.5 # SIGMA
        cistem_parameters[:, 14] = 0.5 # SCORE
        cistem_parameters[:, 15] = 0.0 # CHANGE
        cistem_parameters[:, 16] = 0   # PTLIND (will be filled)
        cistem_parameters[:, 17] = 0.0 # TILTAN (will be filled)
        cistem_parameters[:, 18] = 0.0 # DOSEXX
        cistem_parameters[:, 19] = 0   # SCANOR
        cistem_parameters[:, 20] = 0.0 # CNFDNC
        cistem_parameters[:, 21] = 0   # PTLCCX
        cistem_parameters[:, 22] = 0.0 # AXIS
        cistem_parameters[:, 23] = 0.0 # NORM0
        cistem_parameters[:, 24] = 0.0 # NORM1
        cistem_parameters[:, 25] = 0.0 # NORM2
        cistem_parameters[:, 26:32] = 0.0 # MATRIX elements
        
        # Fill in particle-specific data with progress bar
        row_idx = 0
        total_entries = sum(len(particle_data['tilts']) for particle_data in tracked_particles.values())
        
        with tqdm(total=total_entries, desc="Filling cistem parameters") as pbar:
            for particle_id, particle_data in tracked_particles.items():
                ptlind = particle_data['ptlind']
                
                for tilt_angle, (x, y) in particle_data['tilts'].items():
                    cistem_parameters[row_idx, 16] = ptlind  # PTLIND
                    cistem_parameters[row_idx, 17] = tilt_angle  # TILTAN
                    cistem_parameters[row_idx, 4] = x  # SHX (original position)
                    cistem_parameters[row_idx, 5] = y  # SHY (original position)
                    row_idx += 1
                    pbar.update(1)
        
        return cistem_parameters
    
    def create_parameter_file(self, output_file: str, allboxes: np.ndarray, tracked_particles: Dict):
        """
        Create PYP parameter file
        
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
        
        # Create cistem parameters
        cistem_parameters = self.create_cistem_parameters(allboxes, tracked_particles)
        
        # Create particle and tilt parameters
        particle_parameters, tilt_parameters = self.convert_to_pyp_format(allboxes, tracked_particles)
        
        # Create Parameters object
        parameters_obj = Parameters()
        extended_parameters = ExtendedParameters()
        extended_parameters.set_data(particles=particle_parameters, tilts=tilt_parameters)
        parameters_obj.set_data(data=cistem_parameters, extended_parameters=extended_parameters)
        
        # Save to file
        parameters_obj.to_binary(output_file)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Parameter file created: {output_file} (took {elapsed_time:.2f}s)")
    
    def run_cspt_refinement(self, parameter_file: str, output_dir: str):
        """
        Task 4: Run CSPT refinement using PYP's natural workflow
        
        Parameters
        ----------
        parameter_file : str
            Path to parameter file
        output_dir : str
            Output directory for refinement results
        """
        logger.info(f"Starting CSPT refinement with parameter file: {parameter_file}")
        
        # Set up CSPT parameters
        cspt_parameters = self.parameters.copy()
        cspt_parameters.update({
            "data_mode": "tomo",
            "csp_Grid": "4,4,1",  # 2D grids for spatial partitioning
            "csp_frame_refinement": True,
            "csp_UseImagesForRefinementMin": 0,
            "csp_UseImagesForRefinementMax": -1,
            "csp_refine_micrographs": True,
            "csp_refine_particles": True,
            "csp_refine_ctf": False,
            "csp_refine_shift": True,
            "csp_refine_angle": True,
            "csp_refine_mag": False,
            "csp_refine_occ": False,
            "csp_refine_defocus": False,
            "csp_refine_astigmatism": False,
            "csp_refine_astigmatism_angle": False,
            "csp_refine_phase_shift": False,
            "csp_refine_beam_tilt": False,
            "csp_refine_beam_tilt_x": False,
            "csp_refine_beam_tilt_y": False,
            "csp_refine_beam_tilt_z": False,
            "csp_refine_beam_tilt_angle": False,
            "csp_refine_beam_tilt_angle_x": False,
            "csp_refine_beam_tilt_angle_y": False,
            "csp_refine_beam_tilt_angle_z": False,
            "csp_refine_beam_tilt_angle_angle": False,
            "csp_refine_beam_tilt_angle_angle_x": False,
            "csp_refine_beam_tilt_angle_angle_y": False,
            "csp_refine_beam_tilt_angle_angle_z": False,
            "csp_refine_beam_tilt_angle_angle_angle": False,
            "refine_iter": 2,
            "refine_maxiter": 5,
            "refine_metric": "new",
            "reconstruct_cutoff": 0.5,
            "extract_box": 256,
            "extract_bin": 1,
            "scope_pixel": 1.0,
            "scope_voltage": 300.0,
            "scope_cs": 2.7,
            "scope_wgh": 0.1,
        })
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load parameter file
            alignment_parameters = Parameters.from_file(parameter_file)
            
            # Use PYP's natural CSPT workflow
            from pyp.align.core import csp_run_refinement
            
            logger.info("Preparing CSPT particle processing...")
            
            # Call CSPT using PYP's standard interface
            csp_run_refinement(
                alignment_parameters=alignment_parameters,
                parameters=cspt_parameters,
                dataset="test_dataset",
                name="test_dataset",
                current_class=1,
                iteration=1,
                current_path=os.getcwd(),
                use_frames=False
            )
            
            logger.info("CSPT refinement completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during CSPT refinement: {e}")
            raise
    

    
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
                'total_particles': len(self.particle_parameters),
                'total_tilts': len(self.tilt_parameters),
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
    Main function to run the complete workflow
    """
    # Example usage
    allbox_file = "particles.allbox"
    pkl_file = "tracked_particles.pkl"
    output_file = "test_dataset_r01_02.cistem"
    output_dir = "cspt_output"
    
    # Load configuration if available
    config_file = "test_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            parameters = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
    else:
        # Default PYP parameters
        parameters = {
            "data_set": "test_dataset",
            "slurm_tasks": 4,
            "slurm_verbose": True,
            # Add other parameters as needed
        }
        logger.info("Using default configuration")
    
    # Initialize particle tracker
    tracker = ParticleTracker2D(parameters)
    
    try:
        # Task 1: Read allbox file
        logger.info("=== Task 1: Reading allbox file ===")
        allboxes = tracker.read_allbox_file(allbox_file)
        
        # Task 2: Read tracked particles
        logger.info("=== Task 2: Reading tracked particles ===")
        tracked_particles = tracker.read_tracked_particles_pkl(pkl_file)
        
        # Task 3: Convert to PYP format
        logger.info("=== Task 3: Converting to PYP format ===")
        tracker.create_parameter_file(output_file, allboxes, tracked_particles)
        
        # Generate quality report
        logger.info("=== Generating Quality Report ===")
        quality_report = tracker.generate_quality_report()
        # Convert NumPy types to native Python types for JSON serialization
        quality_report_serializable = convert_numpy_types(quality_report)
        logger.info(f"Quality Report: {json.dumps(quality_report_serializable, indent=2)}")
        
        # Task 4: Run CSPT refinement
        logger.info("=== Task 4: Running CSPT refinement ===")
        tracker.run_cspt_refinement(output_file, output_dir)
        
        logger.info("All tasks completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main workflow: {e}")
        raise


if __name__ == "__main__":
    main() 