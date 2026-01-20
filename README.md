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

### Initial window

![image](./theme/UI-Landing.png "GUI")

The initial window (above) shows two blank image viewers. These will contiaon the distorted (left) and the control (right) images. The top row is for general functions that are applied to both viewers. The bottom row is specific operations that can be applied to the individual viewers. The top tab (not shown) has "File" and "Apply". These are discussed below.

### Importing data

![image](./theme/Import-Data.png "GUI")

The "File" tab provides the ability to open data. You will select either 2D or 3D data to import. The differences are minor between the two, but it determines what the code expects from the inputs. Upon clicking the import button, the above interface will be displayed (the 2D version is shown). You are required to select a distorted image and a control image, but the other inputs are optional. The notes underneath the inputs provide more detail about what the code expects. **NOTE: if you have different resolutions between your two data modes (high res BSE, low res EBSD for example) you will need to specify the pixel resolutions in order to correct for the disparity.** If the resolutions are the same, these entries can be left blank.

For 3D data, the distorted file must be a dream3d file and the control image must be the first image of all control images for the volume (no other files in the folder containing the control images, please).

Once you click "Open", a new interface will pop up (below). This provides a summary of the data read in by the code. Here you can opt to flip the control images (useful for TriBeam datasets, this just rotates the control images 180 degrees) and/or crop the control images to a specific ROI. If cropping is selected, a final UI will show that allows you to drag over an ROI selection. The two continue buttons indicate whether or not rescaling should be applied. As stated above, having different resolutions of data WILL NOT WORK. By rescaling the data, the disparity in resolution is removed and the control data is downsampled to match that of the distorted resolution. This is performed automaticaaly using `skimage.transform.rescale` if selected. YOU MUST SELECT THE OPTION TO RESCALE THE DATA IF DESIRED, RESCALING IS NOT THE DEFAULT.

For EBSD data, the UI can export an ANG file. However, this means you must import the data as an ANG file (it uses the input as a template). Hopefully this can be removed in the future.

![image](./theme/Data-Prep.png "GUI")

At this point you are ready to select control points!

### Selecting control points

Simply click on either image to place a reference point. Make sure that the order of the points is consistent between the two viewers (point 1 on distorted should connect to point 1 on control). Remove points by right cicking on them. The points can be cleared for the current pair of images using the "Clear Points" buttons. The view of the distorted data (typically EBSD data) can be changed using the "EBSD mode" dropdown under the distorted viewer. This allows you to view confidence index, image quality, IPF, color, etc. The "Zoom" dropdown lets you zoom into either image. The scroll bars next to the viewers allow you to move around the image. Currently the scroll bars are the only way to move around.

For 3D data, the "Slice number" dropdown allows you to scroll through the volume and select control points for various slices. A few things to note:
- If only one slice has control points, the same transformation is applied to all slices.
- If multiple slices have control points, the transformation is interpolated between the two closest slices with control points.
- If the first and/or last slice does not have control points, the closest slice with control points in copied to all slices needing extrapolation.
- Example: 10 slices, control points on slices 2 and 7. Slices 1 and 2 use the transformation from slice 2. Slices 3-6 are interpolated transformations from slices 2 and 7. Slices 7-10 use the transformation from slice 2.

### Viewing the alignment

Under the "View transform (current image)" tab, you can select either a TPS or affine-only TPS to view the corrected data in an interactive viewer. "View transform (image stack)" tab provides the same for the entire image stack.

### Saving the data

Under the "File" tab, "Export 2D" will compute the alignment solution and save the corrected data. The first window will be to save the disorted data (corrected). The second window is for saving the control data (press cancel if you do not want to save either one). The outputs will match what one sees in the interaview view. "Export 3D" behaves similar, but it creates a new Dream3D file (based on the user input of the save location) and applies the correction to the data within the new Dream3D file.

**NOTE**: When exporting the data as an image (tif, tiff, png), if one wanted to apply alignment to Confidence Index and Image Quality, one would have to run "Export 2D" twice, each time with a different EBSD mode selected from the dropdown. **The data type is preserved in the saved image, this means that the preview of the image might look wrong, but reading in the data (python, FIJI, etc.) will show the correct data.** For integer data types, the alignemt requires that they go to a float, so the output transforms the float back into the target data type.

**NOTE:** EBSD .ang file exporting is now supported but is not the default. To use this, change the file type in the save as window.


## Other files

- `Outputs.py` Code for saving an ANG file.
- `Inputs.py` Code for reading in dream3d files, images, image stacks, ang files, h5 files, etc...
- `warping.py` Code for converting a set of control point into a transformation between two images (or two stacks) and also applying the transformation.
- `tps.py` Custom class for the Thin-Plate Spline transform.
- `InteractiveView.py` this permits interactive viewing of the solutions.
- "test_data" contains data one can use to test the UI.
- "theme" this contains images and style files for the tkinter GUI and the README file.
- `SIFT.py` is for future plans.

## Future plans

- Adding h5 support for 2D EBSD data
- Adding cropping/rescaling functionality within the UI
- Planning to incorporate automatic control point detection algorithms
- Need to integrate the ability to have multiple control images for each distorted image (imagine toggling between BSE and SE images)
- Plenty of other things...
