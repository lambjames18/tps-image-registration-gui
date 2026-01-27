# Distortion correction for EBSD/BSE images

Contains python files for correcting distorted EBSD images using reference BSE images.

**Disclaimer: This codebase is under development. Although I try to update the README when changes are made, this does not always happen in a timely manner. Please post an issue if something is not working.**


## Usage

To download the code, either run `git clone git@github.com:lambjames18/EBSD-Correction.git` (assuming git is installed) or download the zip file of the repository and unpack it. Once it is downloaded, move into the directory (`cd EBSD-Correction`) and simply run `python ui.py`. The conda environment used during development can be recreated using the `create_env.bat` file (Windows only). This will create a conda environment named "ebsd_correction" with all required packages installed. Note that the pytorch installation line may need to be modified depending on your system and whether or not you have a compatible NVIDIA GPU. See https://pytorch.org/get-started/locally/ for more information.

For information about miniconda (the lightweight command line version of anaconda) see https://docs.conda.io/en/latest/miniconda.html. The environment (named "align" in the command above) will need to be activated in order to run the code. Alternatively, any python interpreter can be used as long as the following packages are installed on your computer:

- `python >= 3.8`
- `numpy`
- `matplotlib`
- `h5py`
- `imageio`
- `scipy`
- `scikit-learn`
- `scikit-image`
- `tifffile`
- `pytorch` (for automatic distortion correction)
- `torchvision` (for automatic distortion correction)
- `kornia` (for automatic distortion correction)

### Tutorial

![image](./theme/GUI-main.png "GUI")

![image](./theme/GUI-points.png "GUI")

![image](./theme/GUI-preview.png "GUI")
