#!/usr/bin/env bash
# One-time setup for a fresh Ubuntu 24.04 VPS.
# Run from inside the quant_trader folder:   bash deploy/setup_vps.sh
set -euo pipefail

echo ">>> Updating the system and installing Python tools ..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip unzip nano ca-certificates curl ufw

echo ">>> Setting the clock to New York time (US market hours) ..."
sudo timedatectl set-timezone America/New_York

echo ">>> Turning on the firewall (only SSH allowed in) ..."
sudo ufw allow OpenSSH
sudo ufw --force enable

echo ">>> Creating the Python environment and installing packages ..."
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
mkdir -p data logs reports models

if [ ! -f .env ]; then
  cp .env.example .env
fi
chmod 600 .env

echo ""
echo "This server's public IP address (for the Trading 212 key's IP restriction):"
curl -s -4 https://ifconfig.me || echo "  (could not look it up - see your VPS provider's dashboard)"
echo ""
echo ""
echo "Done!  Next steps:"
echo "  1. Paste your Trading 212 API key into .env:   nano .env"
echo "  2. Test the connection:                        .venv/bin/python check_connection.py"
