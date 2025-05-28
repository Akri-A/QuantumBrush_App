#!/bin/bash
# QuantumBrush Setup Script
# This script sets up Java and Python dependencies for QuantumBrush

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

# Check Java version
check_java_version() {
    if command -v java &> /dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
        if [ "$JAVA_VERSION" -ge 11 ] 2>/dev/null; then
            print_success "Java $JAVA_VERSION is installed and meets requirements"
            return 0
        else
            print_warning "Java $JAVA_VERSION is installed but version 11+ is required"
            return 1
        fi
    else
        print_warning "Java is not installed"
        return 1
    fi
}

# Install Homebrew if not present
install_homebrew() {
    if ! command -v brew &> /dev/null; then
        print_step "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon Macs
        if [ -f "/opt/homebrew/bin/brew" ]; then
            export PATH="/opt/homebrew/bin:$PATH"
            echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
            echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.bash_profile
        fi
        
        print_success "Homebrew installed successfully"
    else
        print_success "Homebrew is already installed"
    fi
}

# Install Java via Homebrew
install_java() {
    print_step "Installing OpenJDK via Homebrew..."
    
    # Install OpenJDK
    if brew install openjdk; then
        print_success "OpenJDK installed successfully"
        
        # Link it so it's available system-wide
        print_step "Linking OpenJDK for system-wide use..."
        sudo ln -sfn $(brew --prefix)/opt/openjdk/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk
        
        # Add to PATH
        echo 'export PATH="$(brew --prefix)/opt/openjdk/bin:$PATH"' >> ~/.zshrc
        echo 'export PATH="$(brew --prefix)/opt/openjdk/bin:$PATH"' >> ~/.bash_profile
        export PATH="$(brew --prefix)/opt/openjdk/bin:$PATH"
        
        print_success "OpenJDK linked and added to PATH"
    else
        print_error "Failed to install OpenJDK"
        return 1
    fi
    
    # Verify installation
    if check_java_version; then
        return 0
    else
        print_error "Java installation verification failed"
        return 1
    fi
}

# Auto-install Java on macOS
setup_java_macos() {
    print_step "Setting up Java on macOS..."
    
    if check_java_version; then
        return 0
    fi
    
    # Install Homebrew if needed
    install_homebrew
    
    # Install Java
    install_java
}

# Auto-install Java on Linux
setup_java_linux() {
    print_step "Setting up Java on Linux..."
    
    if check_java_version; then
        return 0
    fi
    
    print_step "Installing OpenJDK..."
    
    if command -v apt &> /dev/null; then
        # Ubuntu/Debian
        sudo apt update && sudo apt install -y openjdk-17-jdk
    elif command -v dnf &> /dev/null; then
        # Fedora/RHEL
        sudo dnf install -y java-17-openjdk-devel
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        sudo pacman -S --noconfirm jdk-openjdk
    elif command -v zypper &> /dev/null; then
        # openSUSE
        sudo zypper install -y java-17-openjdk-devel
    else
        print_error "Could not detect package manager. Please install Java manually."
        return 1
    fi
    
    # Verify installation
    if check_java_version; then
        print_success "Java installed successfully"
        return 0
    else
        print_error "Java installation verification failed"
        return 1
    fi
}

# Check for conda installation
check_conda_installation() {
    print_step "Checking for Conda installation..."
    
    # Check if conda command exists
    if command -v conda &> /dev/null; then
        CONDA_PATH=$(which conda)
        print_success "Found existing Conda installation: $CONDA_PATH"
        return 0
    fi
    
    # Check common installation paths
    COMMON_CONDA_PATHS=(
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "$HOME/miniforge3/bin/conda"
        "/opt/miniconda3/bin/conda"
        "/opt/anaconda3/bin/conda"
        "/opt/homebrew/Caskroom/miniconda/base/bin/conda"
        "/opt/homebrew/Caskroom/anaconda/base/bin/conda"
        "/usr/local/miniconda3/bin/conda"
        "/usr/local/anaconda3/bin/conda"
    )
    
    for path in "${COMMON_CONDA_PATHS[@]}"; do
        if [ -f "$path" ]; then
            print_success "Found existing Conda installation: $path"
            # Add to PATH temporarily
            export PATH="$(dirname "$path"):$PATH"
            return 0
        fi
    done
    
    print_warning "No existing Conda installation found"
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

    # Create temporary directory for download
    TEMP_DIR="$HOME/.quantumbrush_temp"
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"

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

    # Clean up installer and temp directory
    cd - > /dev/null
    rm -rf "$TEMP_DIR"
    
    # Initialize conda
    print_step "Initializing Conda..."
    
    # Source conda
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    
    # Initialize for current shells
    "$HOME/miniconda3/bin/conda" init bash 2>/dev/null || true
    "$HOME/miniconda3/bin/conda" init zsh 2>/dev/null || true
    
    print_success "Miniconda installed and initialized successfully"
    return 0
}

# Setup conda environment
setup_conda_environment() {
    print_step "Setting up Python environment for QuantumBrush..."
    
    # Find conda installation
    CONDA_BASE_PATH=""
    
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        CONDA_BASE_PATH="$HOME/miniconda3"
    elif [ -f "$HOME/anaconda3/bin/conda" ]; then
        CONDA_BASE_PATH="$HOME/anaconda3"
    elif command -v conda &> /dev/null; then
        CONDA_BASE_PATH=$(conda info --base 2>/dev/null)
    else
        print_error "Conda not found after installation"
        return 1
    fi
    
    print_step "Using conda at: $CONDA_BASE_PATH"
    
    # Source conda properly
    if [ -f "$CONDA_BASE_PATH/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
        print_success "Conda environment sourced"
    else
        print_error "Conda profile script not found"
        return 1
    fi
    
    # Create or update environment
    CONDA_ENV_NAME="quantumbrush"
    
    if conda env list | grep -q "^$CONDA_ENV_NAME "; then
        print_warning "Environment '$CONDA_ENV_NAME' already exists. Updating..."
        conda env update -n "$CONDA_ENV_NAME" --file - << EOF
name: $CONDA_ENV_NAME
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.11
  - numpy>=2.1.0
  - pillow>=10.0.0
  - matplotlib>=3.7.0
  - scipy>=1.10.0
  - qiskit>=1.0.0
  - pytest>=7.0.0
  - black>=23.0.0
  - pip
EOF
    else
        print_step "Creating new conda environment: $CONDA_ENV_NAME"
        conda create -n "$CONDA_ENV_NAME" python=3.11 -y
        conda install -n "$CONDA_ENV_NAME" -c conda-forge -y \
            numpy \
            pillow \
            matplotlib \
            scipy \
            qiskit \
            pytest \
            black \
    fi
    
    # Get the Python path from the conda environment
    CONDA_PYTHON_PATH=$(conda run -n "$CONDA_ENV_NAME" which python)
    
    print_success "Python environment created: $CONDA_PYTHON_PATH"
    
    # Save Python path to QuantumBrush config
    mkdir -p "config"
    echo "$CONDA_PYTHON_PATH" > "config/python_path.txt"
    
    print_success "Python path saved to QuantumBrush configuration"
}

# Main setup function
main() {
    printf "\n"
    printf "╔══════════════════════════════════════════════════════════════╗\n"
    printf "║                Quantum Brush Setup Script                    ║\n"
    printf "║                                                              ║\n"
    printf "╚══════════════════════════════════════════════════════════════╝\n"
    printf "\n"
    
    print_step "Setting up dependencies for QuantumBrush..."
    
    # Detect OS and setup Java
    OS="$(uname -s)"
    case "${OS}" in
        Darwin*)
            setup_java_macos
            ;;
        Linux*)
            setup_java_linux
            ;;
        *)
            print_error "Unsupported operating system: $OS"
            print_warning "Please install Java 11+ manually"
            ;;
    esac
    
    # Setup Python environment
    if ! check_conda_installation; then
        read -p "Miniconda is required for Python effects. Install it? (Y/n): " -n 1 -r
        echo
        
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            if ! install_miniconda; then
                print_error "Failed to install Miniconda"
                exit 1
            fi
        else
            print_warning "Skipping Miniconda installation. You'll need to configure Python manually."
            exit 0
        fi
    fi
    
    # Set up conda environment
    setup_conda_environment
    
    # Setup complete
    echo
    print_success "Setup completed successfully!"
    echo
    printf "${GREEN}Dependencies installed:${NORMAL}\n"
    printf "  • Java: $(java -version 2>&1 | head -n 1)\n"
    printf "  • Python: $(cat config/python_path.txt 2>/dev/null || echo 'Not configured')\n"
    echo
    printf "${BLUE}You can now run QuantumBrush:${NORMAL}\n"
    printf "  java -jar QuantumBrush.jar\n"
    echo
}

# Run main function
main "$@"
