echo off
echo "Creating conda environment 'ebsd_correction' with required packages..."

call conda create -n ebsd_correction python=3.12 -y
call conda activate ebsd_correction
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy matplotlib h5py scipy scikit-image einops kornia