#!/bin/bash
# ============================================================
# MedGamma Setup Script for Raspberry Pi 5 + Hailo AI HAT+2
# ============================================================
#
# Hardware:
#   - Raspberry Pi 5 (8GB recommended)
#   - Hailo AI HAT+2 (Hailo-10H, 40 TOPS INT8)
#
# This script installs all dependencies for edge inference.
#
# Usage:
#   chmod +x setup_raspberry_pi.sh
#   ./setup_raspberry_pi.sh
#
# ============================================================

set -e

echo "============================================================"
echo "  MedGamma Edge Deployment Setup"
echo "  Raspberry Pi 5 + Hailo-10H (40 TOPS)"
echo "============================================================"
echo ""

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "Warning: Not running on Raspberry Pi"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
else
    MODEL=$(cat /proc/device-tree/model)
    echo "Device: $MODEL"
fi

# Check memory
MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "Memory: ${MEM_GB}GB"

if [ "$MEM_GB" -lt 8 ]; then
    echo "Warning: 8GB RAM recommended for MedGemma inference"
fi

echo ""
echo "[1/6] Updating system..."
sudo apt-get update
sudo apt-get upgrade -y

echo ""
echo "[2/6] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libhdf5-dev \
    git \
    wget

echo ""
echo "[3/6] Creating Python virtual environment..."
cd ~
python3 -m venv medgamma_env
source medgamma_env/bin/activate

echo ""
echo "[4/6] Installing Python packages..."

# Install PyTorch for ARM (CPU only)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    bitsandbytes>=0.42.0 \
    peft>=0.10.0 \
    Pillow>=10.0.0 \
    numpy>=1.24.0 \
    opencv-python-headless \
    huggingface_hub

echo ""
echo "[5/6] Installing Hailo SDK..."

# Check if Hailo SDK is already installed
if python3 -c "from hailo_platform import VDevice" 2>/dev/null; then
    echo "Hailo SDK already installed"
else
    echo ""
    echo "Hailo SDK not found. Please install manually:"
    echo ""
    echo "1. Download Hailo SDK from: https://hailo.ai/developer-zone/"
    echo "2. Follow the installation guide for Raspberry Pi 5"
    echo ""
    echo "Quick install (if you have the .deb package):"
    echo "  sudo dpkg -i hailo-all_*.deb"
    echo "  sudo apt-get install -f"
    echo ""
fi

echo ""
echo "[6/6] Setting up MedGamma..."

# Clone MedGamma if not present
if [ ! -d ~/MedGamma ]; then
    echo "Cloning MedGamma repository..."
    git clone https://github.com/YOUR_USERNAME/MedGamma.git ~/MedGamma
else
    echo "MedGamma directory found at ~/MedGamma"
fi

cd ~/MedGamma

# Create models directory
mkdir -p edge/models

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source ~/medgamma_env/bin/activate"
echo ""
echo "2. Download MedGemma model (requires HuggingFace token):"
echo "   huggingface-cli login"
echo "   python -c \"from transformers import AutoModelForImageTextToText; AutoModelForImageTextToText.from_pretrained('google/medgemma-1.5-4b-it')\""
echo ""
echo "3. Copy your trained checkpoints from Windows:"
echo "   scp -r user@windows-pc:D:/MedGamma/checkpoints/production ~/MedGamma/checkpoints/"
echo ""
echo "4. Copy the Hailo HEF file (if compiled):"
echo "   cp /path/to/sam2_encoder_hailo10h.hef ~/MedGamma/edge/models/"
echo ""
echo "5. Run edge inference:"
echo "   cd ~/MedGamma"
echo "   python edge/edge_inference.py --image test_xray.jpg --task full"
echo ""
echo "============================================================"
