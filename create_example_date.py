#!/usr/bin/env python3
"""
Create example data files for testing the 2D-based tomography workflow
"""

import numpy as np
import pickle
import random

def create_example_allbox_file(filename="particles.allbox", num_particles=100):
    """
    Create example allbox file with particle coordinates
    
    Format: [X-coord, Y-coord, frame=0, tilt-angle]
    """
    print(f"Creating example allbox file: {filename}")
    
    # Generate random particle coordinates
    x_coords = np.random.uniform(100, 900, num_particles)
    y_coords = np.random.uniform(100, 900, num_particles)
    frames = np.zeros(num_particles)  # All frames = 0 (average across frames)
    tilt_angles = np.random.uniform(-60, 60, num_particles)
    
    # Create allbox array
    allboxes = np.column_stack([x_coords, y_coords, frames, tilt_angles])
    
    # Save to file
    np.savetxt(filename, allboxes, fmt='%.2f')
    print(f"Created {filename} with {num_particles} particles")
    
    return allboxes

def create_example_tracked_particles_pkl(filename="tracked_particles.pkl", num_particles=50):
    """
    Create example PKL file with tracked particles
    
    Format: 
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
    """
    print(f"Creating example tracked particles PKL file: {filename}")
    
    # Define tilt angles
    tilt_angles = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    
    tracked_particles = {}
    
    for i in range(num_particles):
        particle_id = f"particle_{i+1}"
        tracked_particles[particle_id] = {}
        
        # Generate base position for this particle
        base_x = np.random.uniform(100, 900)
        base_y = np.random.uniform(100, 900)
        
        # Add some realistic movement across tilts
        for tilt in tilt_angles:
            # Add small random movement to simulate particle drift
            x_offset = np.random.normal(0, 2)  # Small random movement
            y_offset = np.random.normal(0, 2)
            
            x = base_x + x_offset
            y = base_y + y_offset
            
            tracked_particles[particle_id][f"tilt_{tilt}"] = (x, y)
    
    # Save to PKL file
    with open(filename, 'wb') as f:
        pickle.dump(tracked_particles, f)
    
    print(f"Created {filename} with {num_particles} tracked particles across {len(tilt_angles)} tilts")
    
    return tracked_particles

def create_example_config_file(filename="test_config.json"):
    """
    Create example configuration file
    """
    print(f"Creating example config file: {filename}")
    
    config = {
        "data_set": "test_dataset",
        "slurm_tasks": 4,
        "slurm_verbose": True,
        "data_mode": "tomo",
        "csp_Grid": "4,4,1",
        "csp_frame_refinement": True,
        "csp_UseImagesForRefinementMin": 0,
        "csp_UseImagesForRefinementMax": -1,
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
        "tomo_rec_binning": 1,
        "tomo_rec_thickness": 512
    }
    
    import json
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created {filename}")

def main():
    """
    Create all example data files
    """
    print("Creating example data files for testing...")
    
    # Create example allbox file
    allboxes = create_example_allbox_file("particles.allbox", num_particles=100)
    
    # Create example tracked particles PKL file
    tracked_particles = create_example_tracked_particles_pkl("tracked_particles.pkl", num_particles=50)
    
    # Create example config file
    create_example_config_file("test_config.json")
    
    print("\nExample data files created:")
    print("- particles.allbox: Particle coordinates")
    print("- tracked_particles.pkl: Tracked particles with PTLIND assignments")
    print("- test_config.json: Configuration parameters")
    
    print("\nYou can now run: python test.py")

if __name__ == "__main__":
    main() 