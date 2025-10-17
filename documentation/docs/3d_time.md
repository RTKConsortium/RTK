# 3D + time reconstruction

RTK can account for some motion during the acquisition:
- Rigid motion from projection to projection can be accounted for by modifying the [acquisition geometry](Geometry.md).
- Non-rigid (pseudo-)periodic motion (breathing, cardiac, etc.) can be accounted for using the phase of the periodic motion.
- Non-rigid (pseudo-)periodic known motion can be compensated for, or used to guide temporal regularization.

## Phase of periodic motion

When the patient or object moved periodically during the acquisition, a 1D periodic signal describing the motion must be acquired or extracted:
- For beating heart imaging, the electrocardiogram must be recorded during the acquisition and synchronized with it.
- For respiration, a breathing signal can be extracted directly from the projections using the Amsterdam shroud solution of Lambert Zijp described in [[Rit et al, IJROBP, 2021]](https://www.creatis.insa-lyon.fr/~srit/biblio/rit2012.pdf) and implemented in [`rtkamsterdamshroud`](../../applications/rtkamsterdamshroud/README.md).

The phase of that periodic signal is then computed, to determine at which phase each projection was acquired. Phase is commonly measured in radians, with values in $[0,2\pi[$, but in RTK it is normalized to $[0,1[$.

![Signal](./ExternalData/ECG_phase.png){w=800px alt="Phase of an electrocardiogram"}

RTK command-line applications for 3D + time reconstruction have a `–-signal` option to provide a phase file, which should be a text file containing one value in $[0,1[$ per line (see [online example](https://data.kitware.com/api/v1/item/5be99af98d777f2179a2e160/download) of the [ROOSTER page](../../applications/rtkfourdrooster/README.md)). Internally, phase values are then passed to the reconstruction filter as an `std::vector<double>`.

## Frames of a 3D + time reconstruction

3D + time reconstructions are stored as a single 4D image (ie. an `itk::Image<PixelType, 4>`), made of several 3D volumes stacked together, called “frames”. Each frame represents the imaged object at a given phase. The number of frames is chosen by the user, and RTK distributes them evenly along $[0,1[$ (e.g. five frames would represent phases 0, 0.2, 0.4, 0.6 and 0.8).

Phase and frames interact in several ways in RTK:
- [Forward projection of a 4D image](https://www.openrtk.org/Doxygen/classrtk_1_1FourDToProjectionStackImageFilter.html): for each projection, RTK first interpolates linearly between the frames to get a 3D volume at that projection’s phase, then forward projects through that 3D volume.
- [Back projection to a 4D image](https://www.openrtk.org/Doxygen/classrtk_1_1ProjectionStackToFourDImageFilter.html): for each projection, RTK back projects into a 3D volume, then splats that 3D volume linearly.

The forward and back projections are used in all 4D iterative reconstruction methods in RTK.

4D FDK is implemented with a different approach: each frame is reconstructed from a single projection per cycle. Some projections may remain unused.

## Known non-rigid motion

Known motion is passed to RTK as a 3D + time set of deformation vector fields (DVFs). Each DVF is a 3D image with 3D vectors as pixels, i.e. an `itk::Image<itk::Vector<float, 3>, 3>`, and they are stacked together into an `itk::Image<itk::Vector<float, 3>, 4>`.
Like frames, each 3D DVF is associated with a given phase. The value in a 3D DVF at phase $p$ and voxel $v$ is a vector describing the displacement of the voxel $v$ when the object goes from the reference position to the position at phase $p$. The reference position may be one of the phase positions, in which case one of the 3D DVFs is filled with null vectors. Between phase positions, the estimated 3D DVF is obtained by linear interpolation.

## Deformation vector fields (DVFs)

When displaying $\mathrm{DVF}_p$ (e.g. using [vv](https://vv.creatis.insa-lyon.fr/)), the arrow points from the center of voxel $v$ to where that voxel goes when the object goes from the reference position to phase $p$.
But deforming the reference position towards frame $p$ using $\mathrm{DVF}_p$ would require splatting, which requires cumbersome handling of holes and overlaps. Therefore $\mathrm{DVF}_p$ is typically used to deform frame $p$ towards the reference position instead, because that requires only an interpolation. The term “warping” is used in RTK to designate “deforming an image by interpolating in it following a DVF”.


DVFs are used in various ways in RTK:
- To perform a [warped back projection](https://www.openrtk.org/Doxygen/classrtk_1_1WarpProjectionStackToFourDImageFilter.html): for each projection, the back projected volume is computed as if voxels were in their displaced position at the projection’s phase. It is equivalent to a regular back projection followed by warping towards the reference position. This is used in motion-compensated FDK.
- To perform a [warped forward projection](https://www.openrtk.org/Doxygen/classrtk_1_1WarpFourDToProjectionStackImageFilter.html): it assumes the object has undergone a known deformation, and compensates for it during the forward projection, using the inverse deformation. It is equivalent to bending the trajectories of the rays
- Motion-compensated ROOSTER and motion-compensated 4D conjugate gradient perform warped forward and back projections. They reconstruct a 4D image where all frames are deformed to the reference, and then warp them back (using the inverse DVFs) towards their original phase at the end
- In [4D ROOSTER](../../applications/rtkfourdrooster/README.md), DVFs are used only to guide regularization along time: all frames are warped towards the reference frame, temporal regularization is applied, then all frames are warped back (using the inverse DVFs) towards their original phase
