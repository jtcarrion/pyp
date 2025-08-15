# 2D Tilt-Series CSPT Workflow

This document describes how to run the CSPT (Cryo-EM Single Particle Tomography) pipeline using only 2D particle tracks across tilted micrographs, bypassing the requirement for 3D particle coordinates.

## Overview

The CSPT pipeline has been modified to support 2D tilt-series data through the `csp_no_3d_points` parameter. This allows you to:

1. Use 2D particle tracks (x,y coordinates) from each tilted micrograph
2. Bypass 3D specimen bounds calculation and spatial regioning
3. Split refinement per particle instead of per spatial region
4. Run the full reconstruction pipeline using only 2D projection data

## Required Input Data

### 1. 2D Track Data Structure

Your input should be organized as follows:

```python
track_data = {
    'particles': {
        1: {'x_3d': 0.0, 'y_3d': 0.0, 'z_3d': 0.0},  # 3D coords can be zero-filled
        2: {'x_3d': 0.0, 'y_3d': 0.0, 'z_3d': 0.0},
        # ... more particles
    },
    'tilts': {
        0: {'angle': -60.0, 'defocus': 2.5},
        1: {'angle': -45.0, 'defocus': 2.5},
        2: {'angle': 0.0, 'defocus': 2.5},
        # ... more tilts
    },
    'projections': [
        {
            'pind': 1,           # Particle ID
            'tind': 0,           # Tilt index
            'shift_x': 10.5,     # 2D shift from track (pixels)
            'shift_y': -5.2,     # 2D shift from track (pixels)
            'score': 0.8,        # Initial score (optional)
            'occupancy': 1.0,    # Initial occupancy (optional)
        },
        # ... more projections for each particle-tilt pair
    ]
}
```

### 2. Tilt-Series Stack

- **Format**: MRC stack file containing all tilted micrographs
- **Naming**: `{name}_stack.mrc` (e.g., `ts001_stack.mrc`)
- **Structure**: Each slice represents one tilted micrograph in order

### 3. Metadata File

- **Format**: Pickle file containing tilt-series metadata
- **Naming**: `{name}.pkl` (e.g., `ts001.pkl`)
- **Content**: Must contain basic image dimensions and tilt information

## Complete Workflow

### Step 1: Prepare Input Files

```python
from pyp.refine.csp.particle_cspt import create_cistem_from_2d_tracks
from pyp.inout.image import mrc
import pickle

# 1. Create your track data structure
track_data = {
    'particles': {...},  # Your particle data
    'tilts': {...},      # Your tilt data  
    'projections': [...] # Your projection data
}

# 2. Create .cistem parameter files
name = "ts001"
cistem_file, extended_file = create_cistem_from_2d_tracks(
    name=name,
    track_data=track_data,
    parameters=parameters,  # Your PYP parameters
    output_dir="frealign/maps"
)

# 3. Create metadata pickle file
metadata = {
    'tomo': {
        'x': [image_width],
        'y': [image_height], 
        'z': [num_tilts]
    },
    'frames': frame_list  # If using frames
}
with open(f"{name}.pkl", 'wb') as f:
    pickle.dump(metadata, f)

# 4. Create stack file (if not already done)
# mrc.merge_fast(tilt_files, f"{name}_stack.mrc")
```

### Step 2: Configure PYP Parameters

Set these key parameters in your `.pyp_config.toml`:

```toml
# Enable 2D tracks mode
csp_no_3d_points = true

# Data mode must include "tomo" 
data_mode = "tomo"

# Enable frame refinement for per-particle splitting
csp_frame_refinement = true

# Other required parameters
data_set = "your_dataset"
refine_iter = 2
refine_maxiter = 5
extract_box = 256
extract_bin = 2
scope_pixel = 1.35
particle_rad = 100

# CSP-specific parameters
csp_RefineProjectionCutoff = 0.5
csp_UseImagesForRefinementMin = -60
csp_UseImagesForRefinementMax = 60
```

### Step 3: Run CSPT Pipeline

#### Option A: Using csp_split (Recommended)

```python
from pyp_main import csp_split

# This will launch the full CSPT pipeline
csp_split(parameters, iteration=2)
```

#### Option B: Manual Step-by-Step

```python
from pyp.refine.csp.particle_cspt import prepare_particle_cspt, run_reconstruction
from pyp.inout.metadata.frealign_parfile import Parameters

# 1. Load parameter files
name = "ts001"
parameter_file = f"frealign/maps/{name}_r01.cistem"
alignment_parameters = Parameters.from_file(parameter_file)

# 2. Prepare particle CSPT (this is where 2D bypass happens)
split_parx_list = prepare_particle_cspt(
    name=name,
    parameter_file=parameter_file,
    alignment_parameters=alignment_parameters,
    parameters=parameters,  # Must include csp_no_3d_points=True
    grids=[1,1,1],
    use_frames=True
)

# 3. Run reconstruction
run_reconstruction(
    name=name,
    mp=parameters,
    merged_recon_dir="merged_recon",
    output_folder="../output",
    save_stacks=False,
    ref=1,
    iteration=2
)
```

### Step 4: Monitor and Collect Results

The pipeline will create:

- **Intermediate files**: `frealign/scratch/` - temporary processing files
- **Final reconstruction**: `frealign/maps/{name}_r01_02.mrc` - 3D reconstruction
- **Parameter files**: `frealign/maps/{name}_r01_02.cistem` - refined parameters
- **Logs**: `log/` - processing logs and statistics

## Key Implementation Details

### How the 2D Bypass Works

1. **In `prepare_particle_cspt`**: When `csp_no_3d_points=True`:
   - Skips `findSpecimenBounds()` and `divide2regions()`
   - Calls `sort_particles_regions(..., per_particle=True)`
   - Creates one region per particle instead of spatial regions

2. **In `sort_particles_regions`**: When `per_particle=True`:
   - Ignores `corners_squares` and `squaresize` parameters
   - Returns `[[pind1], [pind2], ...]` - one particle per region
   - Does not use `x_position_3d`, `y_position_3d`, `z_position_3d`

3. **In `run_reconstruction`**: 
   - Reads `.cistem` file via `Parameters.from_file()`
   - Uses `get_num_tilts()` and `get_num_frames()` from extended metadata
   - Passes 2D shifts to reconstruct3d for each particle-tilt pair

### Required File Structure

```
project/
├── .pyp_config.toml
├── {name}.pkl                    # Metadata
├── {name}_stack.mrc              # Tilt-series stack
├── frealign/
│   └── maps/
│       ├── {name}_r01.cistem     # Parameter file
│       └── {name}_r01_extended.cistem  # Extended metadata
└── log/                          # Output logs
```

## Troubleshooting

### Common Issues

1. **"Metadata is required" error**:
   - Ensure `{name}.pkl` exists and contains required fields
   - Check that metadata structure matches expected format

2. **"No output parameter file is generated"**:
   - Verify `csp_no_3d_points = true` in parameters
   - Check that `data_mode` contains "tomo"
   - Ensure `csp_frame_refinement = true`

3. **Reconstruction fails**:
   - Verify 2D shifts are in correct units (pixels)
   - Check that tilt angles are consistent
   - Ensure particle IDs and tilt indices are sequential

### Validation Steps

1. **Check parameter file**:
   ```python
   from pyp.inout.metadata.frealign_parfile import Parameters
   par = Parameters.from_file("frealign/maps/ts001_r01.cistem")
   print(f"Projections: {len(par.data)}")
   print(f"Particles: {len(set(par.data[:, 0]))}")
   print(f"Tilts: {len(set(par.data[:, 1]))}")
   ```

2. **Verify 2D shifts**:
   ```python
   shifts_x = par.data[:, 3]  # SHIFT_X column
   shifts_y = par.data[:, 4]  # SHIFT_Y column
   print(f"Shift X range: {shifts_x.min():.2f} to {shifts_x.max():.2f}")
   print(f"Shift Y range: {shifts_y.min():.2f} to {shifts_y.max():.2f}")
   ```

## Performance Considerations

- **Memory usage**: Per-particle splitting can create many small regions
- **Processing time**: More regions = more parallel jobs but smaller individual jobs
- **Scaling**: Consider adjusting `slurm_bundle_size` for optimal performance

## Example Complete Script

```python
#!/usr/bin/env python3

import os
import pickle
import numpy as np
from pyp.refine.csp.particle_cspt import create_cistem_from_2d_tracks
from pyp_main import csp_split
from pyp.system import project_params

def run_2d_tilt_series_cspt():
    """Complete workflow for 2D tilt-series CSPT"""
    
    # 1. Load parameters
    parameters = project_params.load_pyp_parameters()
    
    # 2. Enable 2D tracks mode
    parameters["csp_no_3d_points"] = True
    parameters["csp_frame_refinement"] = True
    parameters["data_mode"] = "tomo"
    
    # 3. Create track data (example)
    track_data = create_example_track_data()
    
    # 4. Create .cistem files
    name = "ts001"
    cistem_file, extended_file = create_cistem_from_2d_tracks(
        name, track_data, parameters, "frealign/maps"
    )
    
    # 5. Create metadata
    create_metadata(name, track_data)
    
    # 6. Run CSPT
    csp_split(parameters, iteration=2)

def create_example_track_data():
    """Create example track data structure"""
    # This would be replaced with your actual 2D track data
    return {
        'particles': {i: {'x_3d': 0.0, 'y_3d': 0.0, 'z_3d': 0.0} for i in range(1, 11)},
        'tilts': {i: {'angle': -60 + i*10, 'defocus': 2.5} for i in range(13)},
        'projections': [
            {'pind': p, 'tind': t, 'shift_x': np.random.normal(0, 5), 
             'shift_y': np.random.normal(0, 5), 'score': 0.8, 'occupancy': 1.0}
            for p in range(1, 11) for t in range(13)
        ]
    }

def create_metadata(name, track_data):
    """Create metadata pickle file"""
    metadata = {
        'tomo': {
            'x': [1024],  # Image width
            'y': [1024],  # Image height
            'z': [len(track_data['tilts'])]  # Number of tilts
        }
    }
    with open(f"{name}.pkl", 'wb') as f:
        pickle.dump(metadata, f)

if __name__ == "__main__":
    run_2d_tilt_series_cspt()
```

This workflow enables you to run the full CSPT pipeline using only 2D particle tracks, making it suitable for cases where 3D coordinates are not available or reliable. 