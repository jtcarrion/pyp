#!/usr/bin/env python3
"""
Simple 2D-Based Tomography Test
===============================

This test validates the 2D-based tomography workflow without PYP dependencies.
"""

import os
import json
import numpy as np
import pickle
import logging
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_allbox_file(allbox_file: str) -> np.ndarray:
    """Read allbox file with particle coordinates"""
    logger.info(f"Reading allbox file: {allbox_file}")
    
    if not os.path.exists(allbox_file):
        raise FileNotFoundError(f"Allbox file not found: {allbox_file}")
    
    allboxes = np.loadtxt(allbox_file, ndmin=2)
    logger.info(f"Loaded {len(allboxes)} particles from allbox file")
    return allboxes

def read_tracked_particles_pkl(pkl_file: str) -> dict:
    """Read tracked particles from PKL file"""
    logger.info(f"Reading tracked particles from PKL file: {pkl_file}")
    
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"PKL file not found: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        tracked_particles = pickle.load(f)
    
    logger.info(f"Loaded {len(tracked_particles)} tracked particles")
    return tracked_particles

def process_particles(allboxes: np.ndarray, tracked_particles: dict) -> dict:
    """Process particles and create output data"""
    logger.info("Processing particles...")
    
    # Count total positions
    total_positions = sum(len(tilt_data) for tilt_data in tracked_particles.values())
    logger.info(f"Total positions across all particles: {total_positions}")
    
    # Create particle data
    particle_data = []
    ptlind_counter = 1
    
    with tqdm(total=len(tracked_particles), desc="Processing particles") as pbar:
        for particle_id, tilt_data in tracked_particles.items():
            for tilt_key, (x, y) in tilt_data.items():
                # Extract tilt angle
                try:
                    tilt_angle = float(tilt_key.replace('tilt_', ''))
                except ValueError:
                    continue
                
                # Create particle entry
                particle = {
                    "particle_index": len(particle_data) + 1,
                    "ptlind": ptlind_counter,
                    "x": float(x),
                    "y": float(y),
                    "tilt_angle": tilt_angle,
                    "particle_id": particle_id
                }
                particle_data.append(particle)
            
            ptlind_counter += 1
            pbar.update(1)
    
    # Create summary
    summary = {
        "total_particles": len(tracked_particles),
        "total_positions": total_positions,
        "unique_tilts": len(set(p["tilt_angle"] for p in particle_data)),
        "allbox_particles": len(allboxes)
    }
    
    result = {
        "particles": particle_data,
        "summary": summary,
        "allbox_data": {
            "num_particles": len(allboxes),
            "x_range": [float(np.min(allboxes[:, 0])), float(np.max(allboxes[:, 0]))],
            "y_range": [float(np.min(allboxes[:, 1])), float(np.max(allboxes[:, 1]))],
            "tilt_range": [float(np.min(allboxes[:, 3])), float(np.max(allboxes[:, 3]))]
        }
    }
    
    logger.info(f"Created {len(particle_data)} particle entries")
    logger.info(f"Summary: {summary}")
    
    return result

def main():
    """Main test function"""
    logger.info("=== Starting Simple 2D-Based Tomography Test ===")
    
    # Input files
    allbox_file = "particles.allbox"
    pkl_file = "tracked_particles.pkl"
    output_file = "test_results.json"
    
    try:
        # Step 1: Read allbox file
        logger.info("Step 1: Reading allbox file")
        allboxes = read_allbox_file(allbox_file)
        
        # Step 2: Read tracked particles
        logger.info("Step 2: Reading tracked particles")
        tracked_particles = read_tracked_particles_pkl(pkl_file)
        
        # Step 3: Process particles
        logger.info("Step 3: Processing particles")
        result = process_particles(allboxes, tracked_particles)
        
        # Step 4: Save results
        logger.info("Step 4: Saving results")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"✅ Test completed successfully!")
        logger.info(f"📁 Output file: {output_file}")
        logger.info(f"📊 Processed {result['summary']['total_positions']} positions from {result['summary']['total_particles']} particles")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 