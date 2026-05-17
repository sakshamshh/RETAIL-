#!/bin/bash

echo "============================================="
echo " Installing Auris Edge Node for Intel N100   "
echo "============================================="

# 1. System Updates & Dependencies
echo "[1/4] Installing system dependencies..."
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git libgl1-mesa-glx libglib2.0-0

# 2. Setup Directory and Venv
echo "[2/4] Setting up /opt/auris environment..."
sudo mkdir -p /opt/auris
sudo chown -R $USER:$USER /opt/auris
cd /opt/auris

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# 3. Install Python requirements
echo "[3/4] Installing Python requirements..."
source venv/bin/activate
pip install --upgrade pip
# OpenCV headless is crucial for Ubuntu servers/N100s without full GUI
pip install opencv-python-headless requests pydantic

# 4. Install Systemd Service
echo "[4/4] Installing systemd service..."
sudo cp auris-edge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable auris-edge.service

echo ""
echo "============================================="
echo "✅ Installation Complete!"
echo ""
echo "Next Steps:"
echo "1. Edit /opt/auris/src/config.py with your real RTSP URLs."
echo "2. Start the service: sudo systemctl start auris-edge"
echo "3. View logs: sudo journalctl -u auris-edge -f"
echo "============================================="
