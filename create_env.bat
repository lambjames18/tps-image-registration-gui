@echo off
setlocal enabledelayedexpansion

REM Set color codes for visual appeal
color 0A

cls
echo.
echo ================================================================================
echo                    EBSD Correction Environment Setup
echo ================================================================================
echo.

REM Step 1: Remove existing environment
echo [STEP 1/4] Checking for existing 'ebsd_correction' environment...
echo.
call conda remove -n ebsd_correction --all -y >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] Removed existing environment
) else (
    echo [•] No existing environment found (creating fresh)
)
echo.

REM Step 2: Create new environment
echo [STEP 2/4] Creating conda environment with Python 3.12...
echo.
call conda create -n ebsd_correction python=3.12 -y
if %errorlevel% neq 0 (
    echo [✗] Failed to create environment. Exiting.
    pause
    exit /b 1
)
echo [✓] Environment created successfully
echo.

REM Step 3: Install PyTorch
echo [STEP 3/4] Installing PyTorch and dependencies...
echo.
call conda run -n ebsd_correction pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if %errorlevel% neq 0 (
    echo [✗] Failed to install PyTorch. Exiting.
    pause
    exit /b 1
)
echo [✓] PyTorch installed successfully
echo.

REM Step 4: Install additional packages
echo [STEP 4/4] Installing additional packages...
echo.
call conda run -n ebsd_correction pip install numpy matplotlib h5py scipy scikit-image einops kornia loguru lightning joblib yacs
if %errorlevel% neq 0 (
    echo [✗] Failed to install additional packages. Exiting.
    pause
    exit /b 1
)
echo [✓] All packages installed successfully
echo.

REM Verification step
echo Verifying installed packages...
echo.
call conda run -n ebsd_correction python -c "import torch; import numpy; import matplotlib; import h5py; import scipy; import skimage; import einops; import kornia; import loguru; import pytorch_lightning; print('[✓] All packages verified!')"
if %errorlevel% neq 0 (
    echo [✗] Package verification failed. Exiting.
    pause
    exit /b 1
)
echo.

REM Success message
echo ================================================================================
echo                         SETUP COMPLETE!
echo ================================================================================
echo.
echo Your 'ebsd_correction' environment is ready to use.
echo.
echo To activate the environment, run:
echo   conda activate ebsd_correction
echo.
echo To deactivate the environment, run:
echo   conda deactivate
echo.
echo ================================================================================
echo.
pause