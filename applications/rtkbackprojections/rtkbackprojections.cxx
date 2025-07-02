/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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
#include "rtkJosephBackAttenuatedProjectionImageFilter.h"
#include "rtkZengBackProjectionImageFilter.h"
#ifdef RTK_USE_CUDA
#  include "rtkCudaFDKBackProjectionImageFilter.h"
#  include "rtkCudaBackProjectionImageFilter.h"
#  include "rtkCudaRayCastBackProjectionImageFilter.h"
#endif
#include "rtkCyclicDeformationImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkbackprojections, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;
#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif
  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::flush;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));
  if (args_info.verbose_flag)
    std::cout << " done." << std::endl;

  // Create an empty volume
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkbackprojections>(constantImageSource,
                                                                                            args_info);

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkbackprojections>(reader, args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update())

  OutputImageType::Pointer attenuationMap;
  if (args_info.attenuationmap_given)
  {
    if (args_info.verbose_flag)
      std::cout << "Reading attenuation map " << args_info.attenuationmap_arg << "..." << std::endl;
    // Read an existing image to initialize the attenuation map
    attenuationMap = itk::ReadImage<OutputImageType>(args_info.attenuationmap_arg);
  }

  // Create back projection image filter
  if (args_info.verbose_flag)
    std::cout << "Backprojecting volume..." << std::endl;
  rtk::BackProjectionImageFilter<OutputImageType, OutputImageType>::Pointer bp;

  // In case warp backprojection is used, we create a deformation
  using DVFPixelType = itk::Vector<float, 3>;
  using DeformationType = rtk::CyclicDeformationImageFilter<itk::Image<DVFPixelType, 4>, itk::Image<DVFPixelType, 3>>;
  auto def = DeformationType::New();

  switch (args_info.bp_arg)
  {
    case (bp_arg_VoxelBasedBackProjection):
      bp = rtk::BackProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case (bp_arg_FDKBackProjection):
      bp = rtk::FDKBackProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case (bp_arg_FDKWarpBackProjection):
      if (!args_info.signal_given || !args_info.dvf_given)
      {
        std::cerr << "FDKWarpBackProjection requires input 4D deformation "
                  << "vector field and signal file names" << std::endl;
        return EXIT_FAILURE;
      }
      def->SetInput(itk::ReadImage<DeformationType::InputImageType>(args_info.dvf_arg));
      bp = rtk::FDKWarpBackProjectionImageFilter<OutputImageType, OutputImageType, DeformationType>::New();
      def->SetSignalFilename(args_info.signal_arg);
      dynamic_cast<rtk::FDKWarpBackProjectionImageFilter<OutputImageType, OutputImageType, DeformationType> *>(
        bp.GetPointer())
        ->SetDeformation(def);
      break;
    case (bp_arg_Joseph):
      bp = rtk::JosephBackProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case (bp_arg_JosephAttenuated):
      bp = rtk::JosephBackAttenuatedProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case (bp_arg_Zeng):
      bp = rtk::ZengBackProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case (bp_arg_CudaFDKBackProjection):
#ifdef RTK_USE_CUDA
      bp = rtk::CudaFDKBackProjectionImageFilter::New();
#else
      std::cerr << "The program has not been compiled with cuda option" << std::endl;
      return EXIT_FAILURE;
#endif
      break;
    case (bp_arg_CudaBackProjection):
#ifdef RTK_USE_CUDA
      bp = rtk::CudaBackProjectionImageFilter<OutputImageType>::New();
#else
      std::cerr << "The program has not been compiled with cuda option" << std::endl;
      return EXIT_FAILURE;
#endif
      break;
    case (bp_arg_CudaRayCast):
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

  bp->SetInput(constantImageSource->GetOutput());
  bp->SetInput(1, reader->GetOutput());
  if (args_info.attenuationmap_given)
    bp->SetInput(2, attenuationMap);
  if (args_info.sigmazero_given && args_info.bp_arg == bp_arg_Zeng)
    dynamic_cast<rtk::ZengBackProjectionImageFilter<OutputImageType, OutputImageType> *>(bp.GetPointer())
      ->SetSigmaZero(args_info.sigmazero_arg);
  if (args_info.alphapsf_given && args_info.bp_arg == bp_arg_Zeng)
    dynamic_cast<rtk::ZengBackProjectionImageFilter<OutputImageType, OutputImageType> *>(bp.GetPointer())
      ->SetAlpha(args_info.alphapsf_arg);
  bp->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(bp->Update())

  // Write
  if (args_info.verbose_flag)
    std::cout << "Writing... " << std::endl;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(bp->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
