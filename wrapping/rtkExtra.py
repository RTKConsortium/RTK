import itk
from itk import RTK as rtk

# Write a 3D circular projection geometry to a file.
def WriteGeometry(geometry, filename):
    # Create and configure the writer
    writer = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
    writer.SetObject(geometry)
    writer.SetFilename(filename)

    # Write the geometry to file
    writer.WriteFile()

# Read a 3D circular projection geometry from a file.
def ReadGeometry(filename):
    # Create and configure the reader
    reader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    reader.SetFilename(filename)
    reader.GenerateOutputInformation()

    # Return the geometry object
    return reader.GetOutputObject()

# Convert an ITK image to a CUDA image if CUDA is available.
def CudaImageFromImage(img):
    if hasattr(itk, 'CudaImage'):
        img.Update()  # Ensure the image is up to date
        cuda_img = itk.CudaImage[itk.itkCType.GetCTypeForDType(img.dtype), img.ndim].New()
        cuda_img.SetPixelContainer(img.GetPixelContainer())
        cuda_img.CopyInformation(img)
        cuda_img.SetBufferedRegion(img.GetBufferedRegion())
        cuda_img.SetRequestedRegion(img.GetRequestedRegion())
        return cuda_img
    return img  # Return the original image if CUDA is not available