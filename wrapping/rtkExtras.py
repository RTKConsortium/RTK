import itk
from itk import RTK as rtk

# Write a 3D circular projection geometry to a file.
def write_geometry(geometry, filename):
  # Create and configure the writer
  writer = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
  writer.SetObject(geometry)
  writer.SetFilename(filename)

  # Write the geometry to file
  writer.WriteFile()

# Read a 3D circular projection geometry from a file.
def read_geometry(filename):
  # Create and configure the reader
  reader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
  reader.SetFilename(filename)
  reader.GenerateOutputInformation()

  # Return the geometry object
  return reader.GetOutputObject()