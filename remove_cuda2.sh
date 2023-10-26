#!/bin/bash

echo "Starting CUDA removal..."

# Remove the NVIDIA CUDA Toolkit
sudo apt-get --purge remove cuda
sudo apt autoremove
sudo apt autoclean

# Remove NVIDIA drivers
sudo apt-get --purge remove 'nvidia*'
sudo apt-get autoremove

# Remove CUDA directories
sudo rm -rf /usr/local/cuda*

echo "CUDA removal process complete. You might want to reboot your system for changes to take effect."


# Additional checks after removal

# Check for NVIDIA-related packages
echo "Checking for NVIDIA-related packages..."
dpkg -l | grep nvidia

# Check for CUDA directories
echo "Checking for CUDA directories..."
ls /usr/local | grep cuda

# Check for NVIDIA kernel modules
echo "Checking for NVIDIA kernel modules..."
lsmod | grep nvidia

echo "Review the output above to verify the removal."
