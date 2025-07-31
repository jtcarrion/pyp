#!/bin/bash
"""
Custom PYP Installation Script with Git-Based Approach
=====================================================

This script installs PYP using your custom code modifications instead of
the pre-built Apptainer image from the official repository.

Key differences from standard install-cli:
1. Clones your Git repository instead of downloading pre-built image
2. Builds custom Apptainer container with your modifications
3. Includes your 2D-based tomography workflow
4. Uses local source code instead of remote pre-built image

Usage:
    ./install-custom.sh [--help] [--force] [--skip-build]
"""

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${PYP_INSTALL_DIR:-$HOME/software/pyp-custom-install}"
CONTAINER_NAME="pyp_2d_tomo.sif"
GIT_REPO="${PYP_GIT_REPO:-$SCRIPT_DIR}"
GIT_BRANCH="${PYP_GIT_BRANCH:-main}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

show_help() {
    cat << EOF
Custom PYP Installation Script

This script installs PYP using your custom Git repository instead of the
pre-built Apptainer image.

USAGE:
    ./install-custom.sh [OPTIONS]

OPTIONS:
    --help          Show this help message
    --force         Force reinstallation (overwrite existing)
    --skip-build    Skip container build (use existing)
    --install-dir   Custom installation directory
    --git-repo      Custom Git repository URL
    --git-branch    Custom Git branch

ENVIRONMENT VARIABLES:
    PYP_INSTALL_DIR    Installation directory (default: ~/software/pyp-custom-install)
    PYP_GIT_REPO       Git repository URL (default: current directory)
    PYP_GIT_BRANCH     Git branch to use

EXAMPLES:
    # Standard installation
    ./install-custom.sh

    # Custom installation directory
    PYP_INSTALL_DIR=/custom/path ./install-custom.sh

    # Force reinstallation
    ./install-custom.sh --force

    # Skip container build
    ./install-custom.sh --skip-build
EOF
}

check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check for Apptainer/Singularity
    if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
        print_error "Apptainer or Singularity not found. Please install one of them first."
        exit 1
    fi
    
    # Check for Git
    if ! command -v git &> /dev/null; then
        print_error "Git not found. Please install Git first."
        exit 1
    fi
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3 first."
        exit 1
    fi
    
    print_success "All dependencies satisfied"
}

setup_installation_directory() {
    print_info "Setting up installation directory: $INSTALL_DIR"
    
    if [ -d "$INSTALL_DIR" ] && [ "$FORCE" != "true" ]; then
        print_warning "Installation directory already exists: $INSTALL_DIR"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installation cancelled"
            exit 0
        fi
    fi
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    print_success "Installation directory ready"
}

clone_repository() {
    print_info "Setting up repository from: $GIT_REPO"
    
    if [ -d "pyp" ]; then
        print_warning "Repository directory already exists"
        if [ "$FORCE" = "true" ]; then
            rm -rf pyp
        else
            read -p "Do you want to remove existing repository? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf pyp
            else
                print_info "Using existing repository"
                return
            fi
        fi
    fi
    
    # If using current directory, copy it instead of cloning
    if [ "$GIT_REPO" = "$SCRIPT_DIR" ]; then
        print_info "Copying current directory to pyp/"
        cp -r "$SCRIPT_DIR" pyp
        # Remove the pyp directory from inside pyp to avoid recursion
        rm -rf pyp/pyp
        print_success "Repository copied successfully"
    else
        # Clone repository
        print_info "Cloning Git repository: $GIT_REPO (branch: $GIT_BRANCH)"
        git clone --branch "$GIT_BRANCH" "$GIT_REPO" pyp
        
        if [ $? -eq 0 ]; then
            print_success "Repository cloned successfully"
        else
            print_error "Failed to clone repository"
            exit 1
        fi
    fi
}

build_container() {
    if [ "$SKIP_BUILD" = "true" ]; then
        print_warning "Skipping container build (--skip-build specified)"
        return
    fi
    
    print_info "Building custom Apptainer container..."
    
    cd pyp
    
    # Check if build script exists
    if [ ! -f "build_container.sh" ]; then
        print_error "build_container.sh not found in repository"
        exit 1
    fi
    
    # Make build script executable
    chmod +x build_container.sh
    
    # Check if apptainer directory exists
    if [ ! -d "apptainer" ]; then
        print_error "apptainer directory not found in repository"
        exit 1
    fi
    
    # Check if recipe file exists
    if [ ! -f "apptainer/pyp.def" ]; then
        print_error "apptainer/pyp.def not found in repository"
        exit 1
    fi
    
    # Build container
    print_info "Building container (this may take 30-60 minutes)..."
    ./build_container.sh
    
    if [ $? -eq 0 ]; then
        print_success "Container built successfully"
    else
        print_error "Container build failed"
        exit 1
    fi
    
    cd ..
}

create_launcher_scripts() {
    print_info "Creating launcher scripts..."
    
    # Create bin directory
    mkdir -p bin
    
    # Create main PYP launcher
    cat > bin/pyp << 'EOF'
#!/bin/bash
# Custom PYP launcher with 2D-based tomography support

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_PATH="$SCRIPT_DIR/../pyp/apptainer/pyp_2d_tomo.sif"

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    echo "Please run the installation script first"
    exit 1
fi

# Execute PYP inside container
apptainer exec "$CONTAINER_PATH" python /opt/pyp/pyp_main.py "$@"
EOF

    # Create custom tomography launcher
    cat > bin/pyp-tomo << 'EOF'
#!/bin/bash
# Custom 2D-based tomography workflow launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_PATH="$SCRIPT_DIR/../pyp/apptainer/pyp_2d_tomo.sif"

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    echo "Please run the installation script first"
    exit 1
fi

# Execute custom tomography workflow
apptainer exec "$CONTAINER_PATH" python /opt/pyp/test.py "$@"
EOF

    # Create hybrid workflow launcher
    cat > bin/pyp-hybrid << 'EOF'
#!/bin/bash
# Hybrid workflow launcher (outside + inside container)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_PATH="$SCRIPT_DIR/../pyp/apptainer/pyp_2d_tomo.sif"

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    echo "Please run the installation script first"
    exit 1
fi

# Execute hybrid workflow
python3 "$SCRIPT_DIR/../pyp/test_hybrid.py" --container-path "$CONTAINER_PATH" "$@"
EOF

    # Make launchers executable
    chmod +x bin/pyp bin/pyp-tomo bin/pyp-hybrid
    
    print_success "Launcher scripts created"
}

create_configuration() {
    print_info "Creating configuration files..."
    
    # Create config directory
    mkdir -p config
    
    # Create custom configuration
    cat > config/pyp_config.toml << 'EOF'
# Custom PYP Configuration for 2D-Based Tomography

[pyp]
scratch = "/tmp/pyp_scratch"
binds = ["/nfs", "/cifs", "/data"]

# 2D-based tomography specific settings
[tomo]
data_mode = "tomo"
csp_Grid = "4,4,1"
csp_frame_range = "1,1"
z_position_3d = 0

# CSPT refinement settings
[csp]
enable_constraints = true
use_ptlind = true
spatial_partitioning = true

# Performance settings
[performance]
parallel_processing = true
memory_efficient = true
EOF

    print_success "Configuration files created"
}

create_documentation() {
    print_info "Creating documentation..."
    
    # Create docs directory
    mkdir -p docs
    
    # Create README
    cat > README.md << 'EOF'
# Custom PYP Installation with 2D-Based Tomography

This is a custom installation of PYP that includes 2D-based tomography particle tracking capabilities.

## Features

- **2D-Based Tomography**: Particle tracking across tilt-series images
- **CSPT Refinement**: Constrained Single-Particle Tomography
- **Hybrid Workflow**: Data preparation outside container, refinement inside
- **Custom Launchers**: Specialized scripts for different workflows

## Usage

### Standard PYP Commands
```bash
# Standard PYP functionality
./bin/pyp --help

# Single-particle analysis
./bin/pyp spa_import_data
```

### Custom Tomography Commands
```bash
# 2D-based tomography workflow
./bin/pyp-tomo --help

# Hybrid workflow (recommended)
./bin/pyp-hybrid --full --allbox-file particles.allbox --pkl-file tracked.pkl
```

### Example Workflow
```bash
# 1. Generate example data
python3 pyp/create_example_data.py

# 2. Run hybrid workflow
./bin/pyp-hybrid --full --allbox-file particles.allbox --pkl-file tracked.pkl

# 3. Check results
ls -la results/
```

## Installation

This installation was created using the custom installation script:
```bash
./install-custom.sh
```

## Configuration

Edit `config/pyp_config.toml` to customize settings for your environment.

## Support

For issues with the custom 2D-based tomography features, check the logs in the scratch directory.
EOF

    print_success "Documentation created"
}

test_installation() {
    print_info "Testing installation..."
    
    # Test container
    if [ -f "pyp/apptainer/$CONTAINER_NAME" ]; then
        print_success "Container found: pyp/apptainer/$CONTAINER_NAME"
    else
        print_error "Container not found"
        exit 1
    fi
    
    # Test launchers
    if [ -f "bin/pyp" ] && [ -x "bin/pyp" ]; then
        print_success "Main launcher created: bin/pyp"
    else
        print_error "Main launcher not found or not executable"
        exit 1
    fi
    
    # Test configuration
    if [ -f "config/pyp_config.toml" ]; then
        print_success "Configuration created: config/pyp_config.toml"
    else
        print_error "Configuration not found"
        exit 1
    fi
    
    print_success "Installation test passed"
}

show_completion_message() {
    cat << EOF

🎉 Custom PYP Installation Complete!
====================================

Installation Directory: $INSTALL_DIR
Container: pyp/apptainer/$CONTAINER_NAME

🚀 Available Commands:
  ./bin/pyp          # Standard PYP functionality
  ./bin/pyp-tomo     # 2D-based tomography workflow
  ./bin/pyp-hybrid   # Hybrid workflow (recommended)

📋 Next Steps:
  1. Add to your PATH: export PATH=\$PATH:$INSTALL_DIR/bin
  2. Test installation: ./bin/pyp --help
  3. Generate example data: python3 pyp/create_example_data.py
  4. Run your workflow: ./bin/pyp-hybrid --full --allbox-file particles.allbox --pkl-file tracked.pkl

📚 Documentation: $INSTALL_DIR/README.md
⚙️  Configuration: $INSTALL_DIR/config/pyp_config.toml

Happy processing! 🧬
EOF
}

# Parse command line arguments
FORCE="false"
SKIP_BUILD="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --force)
            FORCE="true"
            shift
            ;;
        --skip-build)
            SKIP_BUILD="true"
            shift
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --git-repo)
            GIT_REPO="$2"
            shift 2
            ;;
        --git-branch)
            GIT_BRANCH="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main installation process
print_info "Starting custom PYP installation..."

check_dependencies
setup_installation_directory
clone_repository
build_container
create_launcher_scripts
create_configuration
create_documentation
test_installation
show_completion_message

print_success "Custom PYP installation completed successfully!" 