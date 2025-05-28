#!/bin/bash
# QuantumBrush Dependencies Setup Script
# This script sets up Java and Python dependencies

# Configuration
CONDA_ENV_NAME="quantumbrush"
REQUIRED_JAVA_VERSION="11"

# Determine if terminal supports colors
if [ -t 1 ]; then
  if command -v tput > /dev/null; then
    ncolors=$(tput colors)
    if [ -n "$ncolors" ] && [ "$ncolors" -ge 8 ]; then
      BOLD="$(tput bold)"
      NORMAL="$(tput sgr0)"
      RED="$(tput setaf 1)"
      GREEN="$(tput setaf 2)"
      YELLOW="$(tput setaf 3)"
      BLUE="$(tput setaf 4)"
    fi
  fi
fi

# Fallback if tput doesn't work
if [ -z "$RED" ]; then
  BOLD=""
  NORMAL=""
  RED=""
  GREEN=""
  YELLOW=""
  BLUE=""
fi

# Functions for output
print_step() {
    printf "${BLUE}[INFO]${NORMAL} %s\n" "$1"
}

print_success() {
    printf "${GREEN}[SUCCESS]${NORMAL} %s\n" "$1"
}

print_warning() {
    printf "${YELLOW}[WARNING]${NORMAL} %s\n" "$1"
}

print_error() {
    printf "${RED}[ERROR]${NORMAL} %s\n" "$1"
}

# Check for Java installation
check_java() {
    print_step "Checking for Java installation..."
    
    if command -v java &> /dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | awk -F '.' '{print $1}')
        if [ -z "$JAVA_VERSION" ]; then
            # Try alternative method to get Java version
            JAVA_VERSION=$(java -version 2>&1 | head -n 1 | awk -F '"' '{print $2}' | awk -F '.' '{print $1}')
        fi
        
        if [ -z "$JAVA_VERSION" ]; then
            print_warning "Could not determine Java version. Java appears to be installed, but version check failed."
            return 0  # Assume it's OK and continue
        fi
        
        if [ "$JAVA_VERSION" -lt "$REQUIRED_JAVA_VERSION" ]; then
            print_warning "Java version $JAVA_VERSION detected, but QuantumBrush requires Java $REQUIRED_JAVA_VERSION or higher."
            return 1
        else
            print_success "Java $JAVA_VERSION detected (meets requirement of Java $REQUIRED_JAVA_VERSION+)"
            return 0
        fi
    else
        print_error "Java not found on this system."
        return 1
    fi
}

# Install Java using Homebrew (macOS)
install_java_macos() {
    print_step "Installing Java on macOS..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        print_warning "Homebrew not found. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon Macs
        if [ -f "/opt/homebrew/bin/brew" ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    fi
    
    # Install Java using Homebrew
    print_step "Installing Java via Homebrew..."
    brew install --cask temurin
    
    # Verify installation
    if command -v java &> /dev/null; then
        print_success "Java installed successfully!"
        java -version
        return 0
    else
        print_error "Java installation failed. Please install manually."
        return 1
    fi
}

# Install Java on Linux
install_java_linux() {
    print_step "Installing Java on Linux..."
    
    # Detect Linux distribution
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    fi
    
    case $OS in
        "Ubuntu"|"Debian GNU/Linux")
            print_step "Installing Java on Ubuntu/Debian..."
            sudo apt update
            sudo apt install -y openjdk-17-jdk
            ;;
        "Fedora"|"Red Hat Enterprise Linux"|"CentOS Linux")
            print_step "Installing Java on Fedora/RHEL/CentOS..."
            sudo dnf install -y java-17-openjdk
            ;;
        "Arch Linux")
            print_step "Installing Java on Arch Linux..."
            sudo pacman -S --noconfirm jdk-openjdk
            ;;
        *)
            print_error "Unsupported Linux distribution: $OS"
            print_warning "Please install Java manually."
            return 1
            ;;
    esac
    
    # Verify installation
    if command -v java &> /dev/null; then
        print_success "Java installed successfully!"
        java -version
        return 0
    else
        print_error "Java installation failed. Please install manually."
        return 1
    fi
}

# Attempt to install Java automatically
auto_install_java() {
    OS="$(uname -s)"
    
    case "${OS}" in
        Darwin*)
            install_java_macos
            ;;
        Linux*)
            install_java_linux
            ;;
        *)
            print_error "Unsupported operating system: $OS"
            print_warning "Please install Java manually from: https://adoptium.net/"
            return 1
            ;;
    esac
}

# Check if conda is already installed
check_conda_installation() {
    print_step "Checking for existing Anaconda/Miniconda installation..."
    
    # Check if conda command exists
    if command -v conda &> /dev/null; then
        CONDA_PATH=$(which conda)
        print_success "Found existing Conda installation: $CONDA_PATH"
        return 0
    fi
    
    # Check common installation paths
    COMMON_CONDA_PATHS=(
        "$HOME/anaconda3/bin/conda"
        "$HOME/miniconda3/bin/conda"
        "$HOME/miniforge3/bin/conda"
        "/opt/anaconda3/bin/conda"
        "/opt/miniconda3/bin/conda"
        "/usr/local/anaconda3/bin/conda"
        "/usr/local/miniconda3/bin/conda"
        "/opt/homebrew/Caskroom/miniconda/base/bin/conda"
        "/opt/homebrew/Caskroom/anaconda/base/bin/conda"
    )
    
    for path in "${COMMON_CONDA_PATHS[@]}"; do
        if [ -f "$path" ]; then
            print_success "Found existing Conda installation: $path"
            # Add to PATH temporarily
            export PATH="$(dirname "$path"):$PATH"
            return 0
        fi
    done
    
    print_warning "No existing Anaconda/Miniconda installation found"
    return 1
}

# Install Miniconda
install_miniconda() {
    print_step "Installing Miniconda..."
    
    # Detect system architecture
    OS=$(uname -s)
    ARCH=$(uname -m)
    
    case "$OS" in
        "Darwin")
            if [ "$ARCH" = "arm64" ]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
                INSTALLER_NAME="Miniconda3-latest-MacOSX-arm64.sh"
            else
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
                INSTALLER_NAME="Miniconda3-latest-MacOSX-x86_64.sh"
            fi
            ;;
        "Linux")
            if [ "$ARCH" = "x86_64" ]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
                INSTALLER_NAME="Miniconda3-latest-Linux-x86_64.sh"
            elif [ "$ARCH" = "aarch64" ]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
                INSTALLER_NAME="Miniconda3-latest-Linux-aarch64.sh"
            else
                print_error "Unsupported Linux architecture: $ARCH"
                return 1
            fi
            ;;
        *)
            print_error "Unsupported operating system: $OS"
            return 1
            ;;
    esac
    
    print_step "Downloading Miniconda for $OS ($ARCH)..."
    
    # Download Miniconda installer
    if command -v curl &> /dev/null; then
        curl -O "$MINICONDA_URL"
    elif command -v wget &> /dev/null; then
        wget "$MINICONDA_URL"
    else
        print_error "Neither curl nor wget found. Please install one of them."
        return 1
    fi
    
    if [ ! -f "$INSTALLER_NAME" ]; then
        print_error "Failed to download Miniconda installer"
        return 1
    fi
    
    print_step "Installing Miniconda..."
    
    # Install Miniconda silently
    bash "$INSTALLER_NAME" -b -p "$HOME/miniconda3"
    
    # Clean up installer
    rm "$INSTALLER_NAME"
    
    # Initialize conda properly
    print_step "Initializing Conda..."
    
    # Source conda first
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        print_warning "Could not find conda.sh in the expected location"
        # Add conda to PATH as a fallback
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
    
    # Initialize for current shells
    if [ -n "$BASH_VERSION" ] && [ -f "$HOME/.bashrc" ]; then
        "$HOME/miniconda3/bin/conda" init bash
        print_success "Conda initialized for bash"
    fi
    
    if [ -n "$ZSH_VERSION" ] && [ -f "$HOME/.zshrc" ]; then
        "$HOME/miniconda3/bin/conda" init zsh
        print_success "Conda initialized for zsh"
    fi
    
    print_success "Miniconda installed and initialized successfully"
    return 0
}

# Create conda environment for QuantumBrush
setup_conda_environment() {
    print_step "Setting up Python environment for QuantumBrush..."
    
    # Initialize conda properly
    CONDA_BASE_PATH=""
    
    # Find conda installation
    if [ -f "$HOME/anaconda3/bin/conda" ]; then
        CONDA_BASE_PATH="$HOME/anaconda3"
    elif [ -f "$HOME/miniconda3/bin/conda" ]; then
        CONDA_BASE_PATH="$HOME/miniconda3"
    elif [ -f "/opt/homebrew/Caskroom/miniconda/base/bin/conda" ]; then
        CONDA_BASE_PATH="/opt/homebrew/Caskroom/miniconda/base"
    elif [ -f "/opt/homebrew/Caskroom/anaconda/base/bin/conda" ]; then
        CONDA_BASE_PATH="/opt/homebrew/Caskroom/anaconda/base"
    elif command -v conda &> /dev/null; then
        CONDA_BASE_PATH=$(conda info --base 2>/dev/null || echo "")
    fi
    
    if [ -z "$CONDA_BASE_PATH" ]; then
        print_error "Conda not found after installation"
        return 1
    fi
    
    print_step "Using conda at: $CONDA_BASE_PATH"
    
    # Source conda properly
    CONDA_SH_PATHS=(
        "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
        "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
        "/opt/homebrew/Caskroom/anaconda/base/etc/profile.d/conda.sh"
    )
    
    CONDA_SH_FOUND=false
    for conda_sh in "${CONDA_SH_PATHS[@]}"; do
        if [ -f "$conda_sh" ]; then
            source "$conda_sh"
            CONDA_SH_FOUND=true
            print_success "Conda environment sourced from: $conda_sh"
            break
        fi
    done
    
    if [ "$CONDA_SH_FOUND" = false ]; then
        print_warning "Could not find conda.sh. Using conda directly from PATH."
        # Make sure conda is in PATH
        if [ -d "$CONDA_BASE_PATH/bin" ]; then
            export PATH="$CONDA_BASE_PATH/bin:$PATH"
        fi
    fi
    
    # Check if environment already exists
    if conda env list 2>/dev/null | grep -q "^$CONDA_ENV_NAME "; then
        print_warning "Environment '$CONDA_ENV_NAME' already exists. Updating..."
    else
        print_step "Creating new conda environment: $CONDA_ENV_NAME"
        conda create -n "$CONDA_ENV_NAME" python=3.11 -y
    fi
    
    # Install packages using conda run (safer than activate)
    print_step "Installing Python packages..."
    conda install -n "$CONDA_ENV_NAME" -c conda-forge -y \
        numpy \
        pillow \
        opencv \
        matplotlib \
        scipy
    
    # Install additional packages with pip
    conda run -n "$CONDA_ENV_NAME" pip install opencv-python
    
    # Get the Python path from the conda environment
    CONDA_PYTHON_PATH=""
    
    # Try to get Python path using conda run
    if command -v conda &> /dev/null; then
        CONDA_PYTHON_PATH=$(conda run -n "$CONDA_ENV_NAME" which python 2>/dev/null || echo "")
    fi
    
    # If that failed, try to construct the path
    if [ -z "$CONDA_PYTHON_PATH" ]; then
        # Try to find the environment path
        ENV_PATH=$(conda env list | grep "$CONDA_ENV_NAME" | awk '{print $2}')
        if [ -n "$ENV_PATH" ] && [ -f "$ENV_PATH/bin/python" ]; then
            CONDA_PYTHON_PATH="$ENV_PATH/bin/python"
        elif [ -d "$CONDA_BASE_PATH/envs/$CONDA_ENV_NAME/bin" ]; then
            CONDA_PYTHON_PATH="$CONDA_BASE_PATH/envs/$CONDA_ENV_NAME/bin/python"
        else
            print_warning "Could not determine Python path automatically."
            # Ask user for Python path
            read -p "Please enter the full path to Python in your conda environment: " CONDA_PYTHON_PATH
        fi
    fi
    
    if [ -n "$CONDA_PYTHON_PATH" ]; then
        print_success "Python environment created: $CONDA_PYTHON_PATH"
        
        # Save Python path to config
        mkdir -p "config"
        echo "$CONDA_PYTHON_PATH" > "config/python_path.txt"
        
        print_success "Python path saved to config/python_path.txt"
    else
        print_error "Failed to determine Python path. You may need to set it manually."
        return 1
    fi
}

# Main function
main() {
    printf "\n"
    printf "╔══════════════════════════════════════════════════════════════╗\n"
    printf "║              Quantum Brush Dependencies Setup                 ║\n"
    printf "║                                                              ║\n"
    printf "╚══════════════════════════════════════════════════════════════╝\n"
    printf "\n"
    
    # Check for Java
    if ! check_java; then
        echo
        print_warning "Java $REQUIRED_JAVA_VERSION+ is required but not found."
        read -p "Do you want to install Java automatically? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if ! auto_install_java; then
                print_error "Automatic Java installation failed. Please install Java manually."
                exit 1
            fi
        else
            print_warning "Skipping Java installation. Please install Java manually."
            print_warning "Visit https://adoptium.net/ to download Java."
            exit 1
        fi
    fi
    
    # Check for conda installation
    if ! check_conda_installation; then
        echo
        print_warning "Miniconda is required for Python effects processing."
        read -p "Do you want to install Miniconda? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if ! install_miniconda; then
                print_error "Failed to install Miniconda"
                exit 1
            fi
        else
            print_warning "Skipping Miniconda installation. You'll need to configure Python manually."
        fi
    fi
    
    # Set up conda environment
    if command -v conda &> /dev/null || [ -f "$HOME/anaconda3/bin/conda" ] || [ -f "$HOME/miniconda3/bin/conda" ] || [ -f "/opt/homebrew/Caskroom/miniconda/base/bin/conda" ]; then
        setup_conda_environment
    fi
    
    # Setup complete
    echo
    print_success "Dependencies setup completed successfully!"
    echo
    printf "${BLUE}Java:${NORMAL}\n"
    echo "  • Version: $(java -version 2>&1 | head -n 1)"
    echo
    printf "${BLUE}Python Environment:${NORMAL}\n"
    echo "  • Environment name: $CONDA_ENV_NAME"
    echo "  • Python path: $(cat config/python_path.txt 2>/dev/null || echo 'Not configured')"
    echo
    printf "${YELLOW}Important:${NORMAL} Please restart your terminal or run:\n"
    echo "  source ~/.bashrc    (for bash)"
    echo "  source ~/.zshrc     (for zsh)"
    echo
    printf "${GREEN}You can now run Quantum Brush! :${NORMAL}\n"
}

# Run main function
main "$@"
