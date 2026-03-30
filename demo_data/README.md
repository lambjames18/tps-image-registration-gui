# Demo data for EBSD distortion correction project.

## Overview
This directory contains demo data files used for testing and demonstrating the EBSD distortion correction algorithms. The data includes EBSD maps, BSE images, and control point files. Two folders are provided: one for 2D data and another for 3D data. **For both 2D and 3D datasets, the EBSD resolution is 1.5 microns/pixel, and the BSE image resolution is 0.52083333 microns/pixel**.

### 2D Data
The 2D data is a collection of Scanning Electron Microscope (SEM) data from an additively manufactured superalloy material. The dataset includes an Electron Backscatter Diffraction (EBSD) map, Backscattered Electron (BSE) and Secondary Electron (SE) images, and corresponding control point files for distortion correction.

- `EBSD.ang`: EBSD map file in .ang format.
- `BSE.tif`: BSE image taken in a Scanning Electron Microscope (SEM).
- `SE.tif`: SE image taken in a SEM.
- `source_pts.txt`: Text file containing source control points for distortion correction.
- `destination_pts.txt`: Text file containing destination control points for distortion correction.

### 3D Data
The 3D data consists of a series of EBSD maps and BSE images acquired from a nickel-based superalloy material using a Laser-FIB-SEM TriBeam system. The dataset includes multiple slices of EBSD maps and corresponding BSE and SE images, along with control point files for each slice.

- `EBSD.dream3d`: The 3D EBSD dataset generated from the DREAM.3D software.
- `EBSD.xdmf`: Corresponding XDMF file for the DREAM.3D file. Allows for easy visualization in software like ParaView.
- `BSE_001.tif` - `BSE_005.tif`: Series of BSE images for each slice in the 3D dataset.
- `SE_001.tif` - `SE_005.tif`: Series of SE images for each slice in the 3D dataset.
- `source_pts.txt`: Text file containing source control points for distortion correction across the 3D dataset. Each point is a 3D coordinate of the form (slice number, x, y).
- `destination_pts.txt`: Text file containing destination control points for distortion correction across the 3D dataset. Each point is a 3D coordinate of the form (slice number, x, y).
