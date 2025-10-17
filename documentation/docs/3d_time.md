# 3D + time reconstruction

RTK can account for some motion during the acquisition:
-	Rigid motion from projection to projection can be accounted for by modifying the acquisition geometry
-	Non-rigid (pseudo-)periodic motion (breathing, cardiac, etc.) can be accounted for using the phase of the periodic motion
-	Non-rigid (pseudo-)periodic known motion can be compensated for, or used to guide temporal regularization

## Phase of periodic motion

When the acquisition contains periodic motion, a 1D periodic signal describing the motion must be acquired or extracted:
-	For beating heart imaging, the electrocardiogram must be recorded during the acquisition and synchronized with it
-	For breathing chest imaging, a breathing signal can be extracted directly from the projections using the Amsterdam shroud solution of Lambert Zijp ([article](https://www.creatis.insa-lyon.fr/~srit/biblio/rit2012.pdf), [documentation](https://docs.openrtk.org/en/latest/applications/rtkamsterdamshroud/README.html))
The phase of that periodic signal is then computed, to determine at which phase each projection was acquired. Phase is commonly measured in radians, with values in $[0;2Pi[$, but in RTK it is normalized to $[0;1[$.

![Signal](./ExternalData/ECG_phase.png){w=800px alt="Phase of an electrocardiogram"}

RTK command-line applications for 3D + time reconstruction have a `–-signal` option to provide a phase file, which should be a text file containing one value in $[0;1[$ per line. Internally, phase values are then passed to the reconstruction filter as an `std::vector<double>`.

## Frames of a 3D + time reconstruction

3D + time reconstructions are stored as a single 4D image (ie. an `itk::Image<PixelType, 4>`), made of several 3D volumes stacked together, called “frames”. Each frame represents the imaged object at a given phase. The number of frames is chosen by the user, and RTK distributes them evenly along $[0;1[$ (eg. five frames would represent phases 0, 0.2, 0.4, 0.6 and 0.8).
Use of phase and frames in RTK
Phase and frames interact in several ways in RTK:
-	Forward projection of a 4D image ([filter](https://www.openrtk.org/Doxygen/classrtk_1_1FourDToProjectionStackImageFilter.html)): for each projection, RTK first interpolates linearly between the frames to get a 3D volume at that projection’s phase, then forward projects through that 3D volume
-	Back projection to a 4D image ([filter](https://www.openrtk.org/Doxygen/classrtk_1_1ProjectionStackToFourDImageFilter.html)): for each projection, RTK back projects into a 3D volume, then splats that 3D volume linearly
The above forward and back projection mechanisms are used in all 4D iterative reconstruction methods in RTK.
4D FDK is implemented with a different approach: each frame is reconstructed from a single projection per cycle. Some projections may remain unused.

## Non-rigid known motion

Known motion is passed to RTK as a 3D + time set of Deformation Vector Fields (DVFs). Each DVF is a 3D image with 3D vectors as pixels, ie. an `itk::Image<itk::Vector<float, 3>, 3>`, and they are stacked together into an `itk::Image<itk::Vector<float, 3>, 4>`.
Like frames, each 3D DVF is associated with a given phase. The value in a 3D DVF at phase $p$ and voxel $v$ is a vector describing the displacement of the voxel $v$ when the object goes from the reference phase $p_r$ to phase $p$. Therefore, one of the 3D DVFs is usually filled with null vectors (the one at the reference phase). In-between phases, the estimated 3D DVF is obtained by linear interpolation.

## Warping with DVFs

When displaying $DVF_p$ (eg. in vv), the arrow points from the center of voxel $v$ to where that voxel goes when the object goes from the reference phase $p_r$ to phase $p$.
But deforming the reference frame towards frame $p$ using $DVF_p$ would require a splat, which negatively affects image quality. Therefore $DVF_p$ is typically used to deform frame $p$ towards the reference frame $p_r$ instead, because that requires only an interpolation. The term “warping” is used in RTK to designate “deforming an image by interpolating in it following a DVF”.
To deform frame $p_r$ towards phase $p$, it is recommended to warp it using the inverse of $DVF_p$, rather than using $DVF_p$ and performing a splat.

## Use of DVFs in RTK

DVFs are used in various ways in RTK:
-	To perform a warped back projection ([filter](https://www.openrtk.org/Doxygen/classrtk_1_1WarpProjectionStackToFourDImageFilter.html)): for each projection, the back projected volume is computed as if voxels were in their displaced position at the projection’s phase. It is equivalent to a regular back projection followed by warping towards the reference phase. This is used in motion-compensated FDK (application, example)
-	To perform a warped forward projection ([filter](https://www.openrtk.org/Doxygen/classrtk_1_1WarpFourDToProjectionStackImageFilter.html)): it assumes the object has undergone a known deformation, and compensates for it during the forward projection, using the inverse deformation. It is equivalent to bending the trajectories of the rays
-	Motion-compensated ROOSTER and motion-compensated 4D conjugate gradient perform warped forward and back projections. They reconstruct a 4D image where all frames are deformed to the reference frame, and then warp them back (using the inverse DVFs) towards their original phase at the end
-	In 4D ROOSTER[filter](https://www.openrtk.org/Doxygen/classrtk_1_1FourDROOSTERConeBeamReconstructionFilter.html), [application](https://docs.openrtk.org/en/latest/applications/rtkfourdrooster/README.html)), DVFs are used only to guide regularization along time: all frames are warped towards the reference frame, temporal regularization is applied, then all frames are warped back (using the inverse DVFs) towards their original phase
