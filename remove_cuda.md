Here are some instructions for removing CUDA.

echo "Starting CUDA removal..."

# UNIX

## Remove the NVIDIA CUDA Toolkit
sudo apt-get --purge remove cuda
sudo apt autoremove
sudo apt autoclean

## Remove CUDA directories
sudo rm -rf /usr/local/cuda*

echo "CUDA removal process complete. You might want to reboot your system for changes to take effect."



## Remove from env:

conda list | grep cuda





# Windows

Open the Windows Control Panel, go to "Programs and Features," find the NVIDIA CUDA Toolkit entry (or entries, if you have multiple versions installed), and uninstall them.




# Other


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
