#!/bin/bash
# Build Apptainer Container with Custom 2D-Based Tomography Code
# =============================================================
#
# This script rebuilds the PYP Apptainer container with your custom code included.

set -e  # Exit on any error

# Configuration
CONTAINER_NAME="pyp_2d_tomo.sif"
RECIPE_FILE="apptainer/pyp.def"
BUILD_DIR="apptainer"

# Get the script directory to handle relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔨 Building Apptainer container with 2D-based tomography code..."

# Check if we're in the right directory
if [ ! -f "$RECIPE_FILE" ]; then
    echo "❌ Error: Recipe file not found at $RECIPE_FILE"
    echo "Current directory: $(pwd)"
    echo "Looking for: $RECIPE_FILE"
    echo "Available files in current directory:"
    ls -la
    echo ""
    echo "Available files in apptainer directory:"
    ls -la apptainer/ 2>/dev/null || echo "apptainer directory not found"
    echo ""
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if Apptainer is available
if ! command -v apptainer &> /dev/null; then
    echo "❌ Error: Apptainer not found. Please install Apptainer first."
    exit 1
fi

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

# Build the container
echo "📦 Building container from recipe: $RECIPE_FILE"
echo "🏗️  This may take 30-60 minutes depending on your system..."

# Build the container from the root directory, specifying the recipe file path
apptainer build --fakeroot "$BUILD_DIR/$CONTAINER_NAME" "$RECIPE_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Container built successfully!"
    echo "📦 Container file: $BUILD_DIR/$CONTAINER_NAME"
    echo ""
    echo "🚀 Usage examples:"
    echo "  # Test the container"
    echo "  apptainer exec $CONTAINER_NAME python /opt/pyp/test.py --help"
    echo ""
    echo "  # Run hybrid workflow"
    echo "  python test_hybrid.py --full --allbox-file particles.allbox --pkl-file tracked.pkl --container-path $BUILD_DIR/$CONTAINER_NAME"
    echo ""
    echo "  # Generate example data"
    echo "  apptainer exec $CONTAINER_NAME python /opt/pyp/create_example_data.py"
else
    echo "❌ Container build failed!"
    exit 1
fi

echo ""
echo "🎉 Container build complete!"
echo "📋 Next steps:"
echo "  1. Test the container: apptainer exec $BUILD_DIR/$CONTAINER_NAME python /opt/pyp/test.py --help"
echo "  2. Generate example data: python create_example_data.py"
echo "  3. Run your workflow: python test_hybrid.py --full --allbox-file particles.allbox --pkl-file tracked.pkl --container-path $BUILD_DIR/$CONTAINER_NAME" 