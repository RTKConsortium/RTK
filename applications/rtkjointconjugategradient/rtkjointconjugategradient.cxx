/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtkjointconjugategradient_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkSignalToInterpolationWeights.h"
#include "rtkReorderProjectionsImageFilter.h"
#include "rtkVectorImageToImageFilter.h"

#ifdef RTK_USE_CUDA
  #include "itkCudaImage.h"
  #include "rtkCudaConstantVolumeSeriesSource.h"
#endif

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkjointconjugategradient, args_info);

  typedef float PixelValueType;
  const unsigned int Dimension = 3;

  typedef itk::VectorImage< PixelValueType, Dimension > DecomposedProjectionType;
  typedef itk::ImageFileReader<DecomposedProjectionType> DecomposedProjectionReaderType;

  typedef itk::VectorImage< PixelValueType, Dimension > MaterialsVolumeType;
  typedef itk::ImageFileReader< MaterialsVolumeType > MaterialsVolumeReaderType;
  typedef itk::ImageFileWriter< MaterialsVolumeType > MaterialsVolumeWriterType;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< PixelValueType, Dimension + 1 > VolumeSeriesType;
  typedef itk::CudaImage< PixelValueType, Dimension > ProjectionStackType;
#else
  typedef itk::Image< PixelValueType, Dimension + 1 > VolumeSeriesType;
  typedef itk::Image< PixelValueType, Dimension > ProjectionStackType;
#endif

  // Projections reader
  DecomposedProjectionReaderType::Pointer projectionsReader = DecomposedProjectionReaderType::New();
  projectionsReader->SetFileName(args_info.projection_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( projectionsReader->UpdateLargestPossibleRegion() )

  const unsigned int NumberOfMaterials = projectionsReader->GetOutput()->GetVectorLength();

  // Geometry
  if(args_info.verbose_flag)
    std::cout << "Reading geometry information from "
              << args_info.geometry_arg
              << "..."
              << std::endl;
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() )

  // Create 4D input. Fill it either with an existing materials volume read from a file or a blank image
  VolumeSeriesType::Pointer input = VolumeSeriesType::New();
  VolumeSeriesType::SizeType inputSize;
  VolumeSeriesType::SpacingType inputSpacing;
  VolumeSeriesType::PointType inputOrigin;
  VolumeSeriesType::DirectionType inputDirection;

  inputSize[Dimension] = projectionsReader->GetOutput()->GetVectorLength();
  inputSpacing[Dimension] = 1;
  inputOrigin[Dimension] = 0;
  inputDirection.SetIdentity();

  if(args_info.input_given || args_info.like_given)
    {
    // Read an existing vector image to initialize the volume
    MaterialsVolumeReaderType::Pointer referenceReader = MaterialsVolumeReaderType::New();
    if (args_info.input_given)
      referenceReader->SetFileName( args_info.input_arg );
    else
      referenceReader->SetFileName( args_info.like_arg );
    TRY_AND_EXIT_ON_ITK_EXCEPTION( referenceReader->UpdateOutputInformation() );

    VolumeSeriesType::IndexType inputIndex;
    inputIndex.Fill(0);
    VolumeSeriesType::RegionType inputRegion;

    // Initialize the 4D image and (carefully) copy the vector image into it
    for (unsigned int dim=0; dim < Dimension; dim++)
      {
      inputSize[dim] = referenceReader->GetOutput()->GetLargestPossibleRegion().GetSize()[dim];
      inputIndex[dim] = referenceReader->GetOutput()->GetLargestPossibleRegion().GetIndex()[dim];
      inputSpacing[dim] = referenceReader->GetOutput()->GetSpacing()[dim];
      inputOrigin[dim] = referenceReader->GetOutput()->GetOrigin()[dim];
      for (unsigned int j=0; j < Dimension; j++)
        inputDirection[dim][j] = referenceReader->GetOutput()->GetDirection()[dim][j];
      }
    inputIndex[Dimension - 1] = 0;
    inputRegion.SetSize(inputSize);
    inputRegion.SetIndex(inputIndex);
    input->SetRegions(inputRegion);
    input->SetSpacing(inputSpacing);
    input->SetOrigin(inputOrigin);
    input->SetDirection(inputDirection);
    input->Allocate();

    if (args_info.input_given)
      {
      // Actual copy
      itk::ImageRegionConstIterator<MaterialsVolumeType> inIt(referenceReader->GetOutput(), referenceReader->GetOutput()->GetLargestPossibleRegion());
      itk::ImageRegionIterator<VolumeSeriesType> fourdIt(input, inputRegion);
      for (unsigned int material=0; material < referenceReader->GetOutput()->GetVectorLength(); material++)
        {
        inIt.GoToBegin();
        while(!inIt.IsAtEnd())
          {
          fourdIt.Set(inIt.Get()[material]);
          ++inIt;
          ++fourdIt;
          }
        }
      }
      else
      {
      input->FillBuffer(0.);
      }
    }
  else
    {
    // Create new empty volume
    typedef rtk::ConstantImageSource< VolumeSeriesType > ConstantImageSourceType;
    ConstantImageSourceType::Pointer source = ConstantImageSourceType::New();

    inputSize.Fill(args_info.dimension_arg[0]);
    for(unsigned int i=0; i<vnl_math_min(args_info.dimension_given, Dimension); i++)
      inputSize[i] = args_info.dimension_arg[i];

    inputSpacing.Fill(args_info.spacing_arg[0]);
    for(unsigned int i=0; i<vnl_math_min(args_info.spacing_given, Dimension); i++)
      inputSpacing[i] = args_info.spacing_arg[i];

    for(unsigned int i=0; i<Dimension; i++)
      inputOrigin[i] = inputSpacing[i] * (inputSize[i]-1) * -0.5;
    for(unsigned int i=0; i<vnl_math_min(args_info.origin_given, Dimension); i++)
      inputOrigin[i] = args_info.origin_arg[i];

    if(args_info.direction_given)
      for(unsigned int i=0; i<Dimension; i++)
        for(unsigned int j=0; j<Dimension; j++)
          inputDirection[i][j] = args_info.direction_arg[i*Dimension+j];
    else
      inputDirection.SetIdentity();

    source->SetOrigin( inputOrigin );
    source->SetSpacing( inputSpacing );
    source->SetDirection( inputDirection );
    source->SetSize( inputSize );
    source->SetConstant( 0. );
    TRY_AND_EXIT_ON_ITK_EXCEPTION( source->UpdateOutputInformation() );
    input = source->GetOutput();
    }

  // Duplicate geometry and transform the N M-vector projections into N*M scalar projections
  // Each material will occupy one frame of the 4D reconstruction, therefore all projections
  // of one material need to have the same phase.
  // Note : the 4D CG filter is optimized when projections with identical phases are packed together

  // Geometry
  unsigned int initialNumberOfProjections = projectionsReader->GetOutput()->GetLargestPossibleRegion().GetSize()[Dimension - 1];
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry = geometryReader->GetOutputObject();
  for (unsigned int material=1; material<NumberOfMaterials; material++)
    {
    for (unsigned int proj=0; proj<initialNumberOfProjections; proj++)
      {
      geometry->AddProjectionInRadians( geometry->GetSourceToIsocenterDistances()[proj],
                                        geometry->GetSourceToDetectorDistances()[proj],
                                        geometry->GetGantryAngles()[proj],
                                        geometry->GetProjectionOffsetsX()[proj],
                                        geometry->GetProjectionOffsetsY()[proj],
                                        geometry->GetOutOfPlaneAngles()[proj],
                                        geometry->GetInPlaneAngles()[proj],
                                        geometry->GetSourceOffsetsX()[proj],
                                        geometry->GetSourceOffsetsY()[proj]);
      }
    }

  // Signal
  std::vector<double> fakeSignal;
  for (unsigned int material=0; material<NumberOfMaterials; material++)
    {
    for (unsigned int proj=0; proj<initialNumberOfProjections; proj++)
      {
      fakeSignal.push_back(round((double)material / (double)NumberOfMaterials * 1000) / 1000);
      }
    }

  // Projections
  typedef rtk::VectorImageToImageFilter<DecomposedProjectionType, ProjectionStackType> VectorProjectionsToProjectionsFilterType;
  VectorProjectionsToProjectionsFilterType::Pointer vproj2proj = VectorProjectionsToProjectionsFilterType::New();
  vproj2proj->SetInput(projectionsReader->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION( vproj2proj->Update() )

  // Release the memory holding the stack of original projections
  projectionsReader->GetOutput()->ReleaseData();

  // Compute the interpolation weights
  rtk::SignalToInterpolationWeights::Pointer signalToInterpolationWeights = rtk::SignalToInterpolationWeights::New();
  signalToInterpolationWeights->SetSignal(fakeSignal);
  signalToInterpolationWeights->SetNumberOfReconstructedFrames(NumberOfMaterials);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( signalToInterpolationWeights->Update() )

  // Set the forward and back projection filters to be used
  typedef rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType> ConjugateGradientFilterType;
  ConjugateGradientFilterType::Pointer conjugategradient = ConjugateGradientFilterType::New();
  conjugategradient->SetForwardProjectionFilter(args_info.fp_arg);
  conjugategradient->SetBackProjectionFilter(args_info.bp_arg);
  conjugategradient->SetInputVolumeSeries(input);
  conjugategradient->SetNumberOfIterations( args_info.niterations_arg );
  conjugategradient->SetCudaConjugateGradient(args_info.cudacg_flag);

  // Set the newly ordered arguments
  conjugategradient->SetInputProjectionStack( vproj2proj->GetOutput() );
  conjugategradient->SetGeometry( geometry );
  conjugategradient->SetWeights(signalToInterpolationWeights->GetOutput());
  conjugategradient->SetSignal(fakeSignal);

  itk::TimeProbe readerProbe;
  if(args_info.time_flag)
    {
    std::cout << "Recording elapsed time... " << std::flush;
    readerProbe.Start();
    }

  TRY_AND_EXIT_ON_ITK_EXCEPTION( conjugategradient->Update() )

  if(args_info.time_flag)
    {
//    conjugategradient->PrintTiming(std::cout);
    readerProbe.Stop();
    std::cout << "It took...  " << readerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;
    }

  // Write
  typedef itk::ImageFileWriter< VolumeSeriesType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( conjugategradient->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
