#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk

def main():
  # Argument parsing
  parser = argparse.ArgumentParser(
      description='Computes expected photon counts from incident spectrum, material attenuations, detector response and material-decomposed projections',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument('--verbose', '-v', action='store_true', help='Verbose execution')
  parser.add_argument('--config', '-', help='Config file', required=False)
  parser.add_argument('--output', '-o', required=True, help='Output file name (high and low energy projections)')
  parser.add_argument('--input', '-i', required=True, help='Material-decomposed projections')
  parser.add_argument('--high', required=True, help='Incident spectrum image, high energy')
  parser.add_argument('--low', required=True, help='Incident spectrum image, low energy')
  parser.add_argument('--detector', '-d', help='Detector response file', required=False)
  parser.add_argument('--attenuations', '-a', required=True, help='Material attenuations file')
  parser.add_argument('--variances', help='Output variances of measured energies, file name', required=False)

  args_info = parser.parse_args()

  PixelValueType = itk.F
  Dimension = 3

  # Type definitions
  DecomposedProjectionType = itk.VectorImage[PixelValueType, Dimension]
  DecomposedProjectionReaderType = itk.ImageFileReader[DecomposedProjectionType]

  DualEnergyProjectionsType = itk.VectorImage[PixelValueType, Dimension]
  DualEnergyProjectionWriterType = itk.ImageFileWriter[DualEnergyProjectionsType]

  IncidentSpectrumImageType = itk.Image[PixelValueType, Dimension]
  IncidentSpectrumReaderType = itk.ImageFileReader[IncidentSpectrumImageType]

  DetectorResponseImageType = itk.Image[PixelValueType, Dimension - 1]
  DetectorResponseReaderType = itk.ImageFileReader[DetectorResponseImageType]

  MaterialAttenuationsImageType = itk.Image[PixelValueType, Dimension - 1]
  MaterialAttenuationsReaderType = itk.ImageFileReader[MaterialAttenuationsImageType]

  # Read all inputs
  decomposedProjectionReader = DecomposedProjectionReaderType.New()
  decomposedProjectionReader.SetFileName(args_info.input)
  decomposedProjectionReader.Update()

  incidentSpectrumReaderHighEnergy = IncidentSpectrumReaderType.New()
  incidentSpectrumReaderHighEnergy.SetFileName(args_info.high)
  incidentSpectrumReaderHighEnergy.Update()

  incidentSpectrumReaderLowEnergy = IncidentSpectrumReaderType.New()
  incidentSpectrumReaderLowEnergy.SetFileName(args_info.low)
  incidentSpectrumReaderLowEnergy.Update()

  materialAttenuationsReader = MaterialAttenuationsReaderType.New()
  materialAttenuationsReader.SetFileName(args_info.attenuations)
  materialAttenuationsReader.Update()

  # Get parameters from the images
  MaximumEnergy = incidentSpectrumReaderHighEnergy.GetOutput().GetLargestPossibleRegion().GetSize(0)

  # If the detector response is given by the user, use it. Otherwise, assume it is included in the
  # incident spectrum, and fill the response with ones
  detectorResponseReader = DetectorResponseReaderType.New()
  detectorImage = None
  if args_info.detector:
      detectorResponseReader.SetFileName(args_info.detector)
      detectorResponseReader.Update()
      detectorImage = detectorResponseReader.GetOutput()
  else:
      detectorSource = rtk.ConstantImageSource[DetectorResponseImageType].New()
      sourceSize = DetectorResponseImageType.SizeType()
      sourceSize[0] = 1
      sourceSize[1] = MaximumEnergy
      detectorSource.SetSize(sourceSize)
      detectorSource.SetConstant(1.0)
      detectorSource.Update()
      detectorImage = detectorSource.GetOutput()

  # Generate a set of zero-filled intensity projections
  dualEnergyProjections = DualEnergyProjectionsType.New()
  dualEnergyProjections.CopyInformation(decomposedProjectionReader.GetOutput())
  dualEnergyProjections.SetVectorLength(2)
  dualEnergyProjections.Allocate()

  # Check that the inputs have the expected size
  if decomposedProjectionReader.GetOutput().GetVectorLength() != 2:
      raise ValueError(f'Decomposed projections image has vector length {decomposedProjectionReader.GetOutput().GetVectorLength()}, should be 2')

  if materialAttenuationsReader.GetOutput().GetLargestPossibleRegion().GetSize()[1] != MaximumEnergy:
      raise ValueError(f'Material attenuations image has {materialAttenuationsReader.GetOutput().GetLargestPossibleRegion().GetSize()[1]} energies, should have {MaximumEnergy}')

  # Create and set the filter
  ForwardModelFilterType = rtk.SpectralForwardModelImageFilter[DecomposedProjectionType, DualEnergyProjectionsType]
  forward = ForwardModelFilterType.New()
  forward.SetInputDecomposedProjections(decomposedProjectionReader.GetOutput())
  forward.SetInputMeasuredProjections(dualEnergyProjections)
  forward.SetInputIncidentSpectrum(incidentSpectrumReaderHighEnergy.GetOutput())
  forward.SetInputSecondIncidentSpectrum(incidentSpectrumReaderLowEnergy.GetOutput())
  forward.SetDetectorResponse(detectorImage)
  forward.SetMaterialAttenuations(materialAttenuationsReader.GetOutput())
  forward.SetIsSpectralCT(False)
  forward.SetComputeVariances(args_info.variances)

  forward.Update()

  # Write output
  writer = DualEnergyProjectionWriterType.New()
  writer.SetInput(forward.GetOutput())
  writer.SetFileName(args_info.output)
  writer.Update()

  # If requested, write the variances
  if args_info.variances:
    itk.WriteImage(forward.GetOutputVariances(), args_info.variances)

if __name__ == '__main__':
    main()
