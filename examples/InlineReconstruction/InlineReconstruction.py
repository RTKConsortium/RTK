import sys
import os
import time
import threading
import itk
from itk import RTK as rtk

# Parameters controlling the geometry of the simulated acquisition
image_type = itk.Image[itk.F, 3]
has_gpu_capability = hasattr(itk, "CudaImage")
if has_gpu_capability:
    cuda_image_type = itk.CudaImage[itk.F, 3]
nproj = 64
sid = 1000
sdd = 1500
arc = 200
spacing = 1
size = 256
origin = -0.5 * (size - 1)
number_of_acquired_projections = 0

# Function for the thread simulating an acquisition: simulates and writes a
# projection every 100 ms.
def acquisition():
    global number_of_acquired_projections
    for i in range(nproj):
        geometry_acq = rtk.ThreeDCircularProjectionGeometry.New()
        geometry_acq.AddProjection(sid, sdd, i * arc / nproj)
        projection = rtk.constant_image_source(
            origin=[origin, origin, 0.0],
            size=[size, size, 1],
            spacing=[spacing] * 3,
            ttype=image_type,
        )
        projection = rtk.shepp_logan_phantom_filter(
            projection,
            geometry=geometry_acq,
            phantom_scale=100,
        )
        itk.imwrite(projection, f"projection_{i:03d}.mha")

        # The code assumes that incrementing an integer is thread safe, which
        # is the case with CPython
        # https://docs.python.org/3/library/threading.html
        number_of_acquired_projections += 1
        time.sleep(0.1)



# Launches the acquisition thread, then waits for new projections and reconstructs inline / on-the-fly
# Launch the simulated acquisition
thread = threading.Thread(target=acquisition)
thread.start()

# Create the (expected) geometry and filenames
geometry_rec = rtk.ThreeDCircularProjectionGeometry.New()
for i in range(nproj):
    geometry_rec.AddProjection(sid, sdd, i * arc / nproj)

# Create the reconstruction pipeline
reader = rtk.ProjectionsReader[image_type].New()
extractor = itk.ExtractImageFilter[image_type, image_type].New(
    Input=reader.GetOutput()
)
if has_gpu_capability:
    parker = rtk.CudaParkerShortScanImageFilter.New(Geometry=geometry_rec)
    reconstruction_source = rtk.ConstantImageSource[cuda_image_type].New(
        Origin=[origin * sid / sdd] * 3,
        Spacing=[spacing * sid / sdd] * 3,
        Size=[size] * 3,
    )
    fdk = rtk.CudaFDKConeBeamReconstructionFilter.New(Geometry=geometry_rec)
else:
    parker = rtk.ParkerShortScanImageFilter[image_type].New(
        Input=extractor.GetOutput(), Geometry=geometry_rec
    )
    reconstruction_source = rtk.ConstantImageSource[image_type].New(
        Origin=[origin * sid / sdd] * 3,
        Spacing=[spacing * sid / sdd] * 3,
        Size=[size] * 3,
    )
    fdk = rtk.FDKConeBeamReconstructionFilter[image_type].New(Geometry=geometry_rec)
fdk.SetInput(0, reconstruction_source.GetOutput())
fdk.SetInput(1, parker.GetOutput())

# Do the online / on-the-fly reconstruction: wait for acquired projections
# and use them as soon as a new one is available
number_of_reconstructed_projections = 0
while number_of_reconstructed_projections != nproj:
    new_projection_available = bool(
        number_of_reconstructed_projections < number_of_acquired_projections
    )
    if not new_projection_available:
        time.sleep(0.01)
    else:
        print(
            f"Processing projection #{number_of_reconstructed_projections}",
            end="\r",
        )
        if number_of_reconstructed_projections == 0:
            # First projection, mimick a stack from one file and prepare extracted region
            projection_file_names = ["projection_000.mha"] * nproj
            reader.SetFileNames(projection_file_names)
            reader.UpdateOutputInformation()
            extracted_region = reader.GetOutput().GetLargestPossibleRegion()
            extracted_region.SetSize(2, 1)
        else:
            # Update file name list with the new projection
            projection_file_names[
                number_of_reconstructed_projections
            ] = f"projection_{number_of_reconstructed_projections:03d}.mha"
            reader.SetFileNames(projection_file_names)

            # Reconnect FDK output to
            reconstructed_image = fdk.GetOutput()
            reconstructed_image.DisconnectPipeline()
            fdk.SetInput(reconstructed_image)

        # Only extract and read the new projection
        extracted_region.SetIndex(2, number_of_reconstructed_projections)
        extractor.SetExtractionRegion(extracted_region)
        extractor.UpdateLargestPossibleRegion()
        if has_gpu_capability:
            projection = cuda_image_type.New()
            projection.SetPixelContainer(extractor.GetOutput().GetPixelContainer())
            projection.CopyInformation(extractor.GetOutput())
            projection.SetBufferedRegion(extractor.GetOutput().GetBufferedRegion())
            projection.SetRequestedRegion(
                extractor.GetOutput().GetRequestedRegion()
            )
            parker.SetInput(projection)
        fdk.Update()
        number_of_reconstructed_projections += 1
thread.join()
writer = itk.ImageFileWriter[image_type].New(
    Input=fdk.GetOutput(), FileName="fdk.mha"
)
writer.Update()
