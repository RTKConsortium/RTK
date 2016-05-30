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

#include "rtkbackprojections_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkFDKBackProjectionImageFilter.h"
#include "rtkFDKWarpBackProjectionImageFilter.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkNormalizedJosephBackProjectionImageFilter.h"
#ifdef RTK_USE_CUDA
#  include "rtkCudaFDKBackProjectionImageFilter.h"
#  include "rtkCudaBackProjectionImageFilter.h"
#  include "rtkCudaRayCastBackProjectionImageFilter.h"
#endif
#include "rtkCyclicDeformationImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkTimeProbe.h>

int main(int argc, char * argv[])
{
  GGO(rtkbackprojections, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
#endif
  // Geometry
  if(args_info.verbose_flag)
    std::cout << "Reading geometry information from "
              << args_info.geometry_arg
              << "..."
              << std::flush;
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() )
  if(args_info.verbose_flag)
    std::cout << " done." << std::endl;

  // Create an empty volume
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkbackprojections>(constantImageSource, args_info);

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkbackprojections>(reader, args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )

  // Create back projection image filter
  if(args_info.verbose_flag)
    std::cout << "Backprojecting volume..." << std::flush;
  itk::TimeProbe bpProbe;
  rtk::BackProjectionImageFilter<OutputImageType, OutputImageType>::Pointer bp;

  // In case warp backprojection is used, we create a deformation
  typedef itk::Vector<float,3> DVFPixelType;
  typedef itk::Image< DVFPixelType, 3 > DVFImageType;
  typedef rtk::CyclicDeformationImageFilter< DVFImageType > DeformationType;
  typedef itk::ImageFileReader<DeformationType::InputImageType> DVFReaderType;
  DVFReaderType::Pointer dvfReader = DVFReaderType::New();
  DeformationType::Pointer def = DeformationType::New();
  def->SetInput(dvfReader->GetOutput());

  switch(args_info.bp_arg)
  {
    case(bp_arg_VoxelBasedBackProjection):
      bp = rtk::BackProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case(bp_arg_FDKBackProjection):
      bp = rtk::FDKBackProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case(bp_arg_FDKWarpBackProjection):
      if(!args_info.signal_given || !args_info.dvf_given)
        {
        std::cerr << "FDKWarpBackProjection requires input 4D deformation "
                  << "vector field and signal file names"
                  << std::endl;
        return EXIT_FAILURE;
        }
      dvfReader->SetFileName(args_info.dvf_arg);
      bp = rtk::FDKWarpBackProjectionImageFilter<OutputImageType, OutputImageType, DeformationType>::New();
      def->SetSignalFilename(args_info.signal_arg);
      dynamic_cast<rtk::FDKWarpBackProjectionImageFilter<OutputImageType, OutputImageType, DeformationType>*>(bp.GetPointer())->SetDeformation(def);
      break;
    case(bp_arg_Joseph):
      bp = rtk::JosephBackProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case(bp_arg_NormalizedJoseph):
      bp = rtk::NormalizedJosephBackProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case(bp_arg_CudaFDKBackProjection):
#ifdef RTK_USE_CUDA
      bp = rtk::CudaFDKBackProjectionImageFilter::New();
#else
      std::cerr << "The program has not been compiled with cuda option" << std::endl;
      return EXIT_FAILURE;
#endif
      break;
    case(bp_arg_CudaBackProjection):
#ifdef RTK_USE_CUDA
      bp = rtk::CudaBackProjectionImageFilter::New();
#else
      std::cerr << "The program has not been compiled with cuda option" << std::endl;
      return EXIT_FAILURE;
#endif
      break;
    case(bp_arg_CudaRayCast):
#ifdef RTK_USE_CUDA
      bp = rtk::CudaRayCastBackProjectionImageFilter::New();
#else
      std::cerr << "The program has not been compiled with cuda option" << std::endl;
      return EXIT_FAILURE;
#endif
      break;
    default:
    std::cerr << "Unhandled --method value." << std::endl;
    return EXIT_FAILURE;
  }

  bp->SetInput( constantImageSource->GetOutput() );
  bp->SetInput( 1, reader->GetOutput() );
  bp->SetGeometry( geometryReader->GetOutputObject() );
  bpProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( bp->Update() )
  bpProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << bpProbe.GetMean() << ' ' << bpProbe.GetUnit()
              << '.' << std::endl;

  // Write
  if(args_info.verbose_flag)
    std::cout << "Writing... " << std::flush;
  itk::TimeProbe writeProbe;
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( bp->GetOutput() );
  writeProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )
  writeProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << writeProbe.GetMean() << ' ' << writeProbe.GetUnit()
              << '.' << std::endl;

  return EXIT_SUCCESS;
}
