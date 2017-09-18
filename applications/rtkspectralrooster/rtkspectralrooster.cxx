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

#include "rtkspectralrooster_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkFourDROOSTERConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkSignalToInterpolationWeights.h"
#include "rtkVectorImageToImageFilter.h"
#include "rtkImageToVectorImageFilter.h"

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
  GGO(rtkspectralrooster, args_info);

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
  VolumeSeriesType::Pointer input;

  typedef rtk::VectorImageToImageFilter<MaterialsVolumeType, VolumeSeriesType> VectorVolumeToVolumeSeriesFilterType;
  VectorVolumeToVolumeSeriesFilterType::Pointer vecVol2VolSeries = VectorVolumeToVolumeSeriesFilterType::New();

  if(args_info.input_given)
    {
    // Using std::cout because itkWarningMacro cannot be used outside a class
    if(args_info.like_given)
      std::cout << "WARNING: Option --like ignored, since option --input was passed" << std::endl;

    MaterialsVolumeReaderType::Pointer referenceReader = MaterialsVolumeReaderType::New();
    referenceReader->SetFileName( args_info.input_arg );
    vecVol2VolSeries->SetInput(referenceReader->GetOutput());
    vecVol2VolSeries->Update();
    input = vecVol2VolSeries->GetOutput();
    }
  else if(args_info.like_given)
    {
    MaterialsVolumeReaderType::Pointer referenceReader = MaterialsVolumeReaderType::New();
    referenceReader->SetFileName( args_info.like_arg );
    vecVol2VolSeries->SetInput(referenceReader->GetOutput());
    vecVol2VolSeries->UpdateOutputInformation();

    typedef rtk::ConstantImageSource< VolumeSeriesType > ConstantImageSourceType;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    constantImageSource->SetInformationFromImage(vecVol2VolSeries->GetOutput());
    constantImageSource->Update();
    input = constantImageSource->GetOutput();
    }
  else
    {
    // Create new empty volume
    typedef rtk::ConstantImageSource< VolumeSeriesType > ConstantImageSourceType;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();

    VolumeSeriesType::SizeType inputSize;
    VolumeSeriesType::SpacingType inputSpacing;
    VolumeSeriesType::PointType inputOrigin;
    VolumeSeriesType::DirectionType inputDirection;

    inputSize[Dimension] = projectionsReader->GetOutput()->GetVectorLength();
    inputSpacing[Dimension] = 1;
    inputOrigin[Dimension] = 0;
    inputDirection.SetIdentity();

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

    constantImageSource->SetOrigin( inputOrigin );
    constantImageSource->SetSpacing( inputSpacing );
    constantImageSource->SetDirection( inputDirection );
    constantImageSource->SetSize( inputSize );
    constantImageSource->SetConstant( 0. );
    TRY_AND_EXIT_ON_ITK_EXCEPTION( constantImageSource->Update() );
    input = constantImageSource->GetOutput();
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
      geometry->SetCollimationOfLastProjection(geometry->GetCollimationUInf()[proj],
                                               geometry->GetCollimationUSup()[proj],
                                               geometry->GetCollimationVInf()[proj],
                                               geometry->GetCollimationVSup()[proj]);
      }
    }

  // Signal
  std::vector<double> fakeSignal;
  for (unsigned int material=0; material<NumberOfMaterials; material++)
    {
    for (unsigned int proj=0; proj<initialNumberOfProjections; proj++)
      {
      fakeSignal.push_back(itk::Math::Round<double, double>((double)material / (double)NumberOfMaterials * 1000) / 1000);
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
  typedef rtk::FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType> ROOSTERFilterType;
  ROOSTERFilterType::Pointer rooster = ROOSTERFilterType::New();
  rooster->SetForwardProjectionFilter(args_info.fp_arg);
  rooster->SetBackProjectionFilter(args_info.bp_arg);
  rooster->SetInputVolumeSeries(input);
  rooster->SetCG_iterations( args_info.cgiter_arg );
  rooster->SetMainLoop_iterations( args_info.niter_arg );
  rooster->SetCudaConjugateGradient(args_info.cudacg_flag);
  rooster->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  // Set the newly ordered arguments
  rooster->SetInputProjectionStack( vproj2proj->GetOutput() );
  rooster->SetGeometry( geometry );
  rooster->SetWeights(signalToInterpolationWeights->GetOutput());
  rooster->SetSignal(fakeSignal);

  // For each optional regularization step, set whether or not
  // it should be performed, and provide the necessary inputs

  // Positivity
  if (args_info.nopositivity_flag)
    rooster->SetPerformPositivity(false);
  else
    rooster->SetPerformPositivity(true);

  // No motion mask is used, since there is no motion
  rooster->SetPerformMotionMask(false);

  // Spatial TV
  if (args_info.gamma_space_given)
    {
    rooster->SetGammaTVSpace(args_info.gamma_space_arg);
    rooster->SetTV_iterations(args_info.tviter_arg);
    rooster->SetPerformTVSpatialDenoising(true);
    }
  else
    rooster->SetPerformTVSpatialDenoising(false);

  // Spatial wavelets
  if (args_info.threshold_given)
    {
    rooster->SetSoftThresholdWavelets(args_info.threshold_arg);
    rooster->SetOrder(args_info.order_arg);
    rooster->SetNumberOfLevels(args_info.levels_arg);
    rooster->SetPerformWaveletsSpatialDenoising(true);
    }
  else
    rooster->SetPerformWaveletsSpatialDenoising(false);

  // Temporal TV
  if (args_info.gamma_time_given)
    {
    rooster->SetGammaTVTime(args_info.gamma_time_arg);
    rooster->SetTV_iterations(args_info.tviter_arg);
    rooster->SetPerformTVTemporalDenoising(true);
    }
  else
    rooster->SetPerformTVTemporalDenoising(false);

  // Temporal L0
  if (args_info.lambda_time_arg)
    {
    rooster->SetLambdaL0Time(args_info.lambda_time_arg);
    rooster->SetL0_iterations(args_info.l0iter_arg);
    rooster->SetPerformL0TemporalDenoising(true);
    }
  else
    rooster->SetPerformL0TemporalDenoising(false);

  // Total nuclear variation
  if (args_info.gamma_tnv_given)
    {
    rooster->SetGammaTNV(args_info.gamma_tnv_arg);
    rooster->SetTV_iterations(args_info.tviter_arg);
    rooster->SetPerformTNVDenoising(true);
    }
  else
    rooster->SetPerformTNVDenoising(false);

  itk::TimeProbe readerProbe;
  if(args_info.time_flag)
    {
    std::cout << "Recording elapsed time... " << std::flush;
    readerProbe.Start();
    }

  TRY_AND_EXIT_ON_ITK_EXCEPTION( rooster->Update() )

  if(args_info.time_flag)
    {
//    rooster->PrintTiming(std::cout);
    readerProbe.Stop();
    std::cout << "It took...  " << readerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;
    }

  // Convert to result to a vector image
  typedef rtk::ImageToVectorImageFilter<VolumeSeriesType, MaterialsVolumeType> VolumeSeriesToVectorVolumeFilterType;
  VolumeSeriesToVectorVolumeFilterType::Pointer volSeries2VecVol = VolumeSeriesToVectorVolumeFilterType::New();
  volSeries2VecVol->SetInput(rooster->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION( volSeries2VecVol->Update() )

  // Write
  typedef itk::ImageFileWriter< MaterialsVolumeType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( volSeries2VecVol->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
