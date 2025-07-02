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

#include "rtkforwardprojections_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkMaximumIntensityProjectionImageFilter.h"
#include "rtkJosephForwardAttenuatedProjectionImageFilter.h"
#include "rtkZengForwardProjectionImageFilter.h"
#ifdef RTK_USE_CUDA
#  include "rtkCudaForwardProjectionImageFilter.h"
#endif

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkforwardprojections, args_info);

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

  // Create a stack of empty projection images
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkforwardprojections>(constantImageSource,
                                                                                               args_info);

  // Adjust size according to geometry
  auto sizeOutput = itk::MakeSize(
    constantImageSource->GetSize()[0], constantImageSource->GetSize()[1], geometry->GetGantryAngles().size());
  constantImageSource->SetSize(sizeOutput);

  // Input reader
  if (args_info.verbose_flag)
    std::cout << "Reading input volume " << args_info.input_arg << "..." << std::endl;
  OutputImageType::Pointer inputVolume;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(inputVolume = itk::ReadImage<OutputImageType>(args_info.input_arg))

  OutputImageType::Pointer attenuationMap;
  if (args_info.attenuationmap_given)
  {
    if (args_info.verbose_flag)
      std::cout << "Reading attenuation map " << args_info.attenuationmap_arg << "..." << std::endl;
    // Read an existing image to initialize the attenuation map
    attenuationMap = itk::ReadImage<OutputImageType>(args_info.attenuationmap_arg);
  }

  using ClipImageType = itk::Image<double, Dimension>;
  ClipImageType::Pointer inferiorClipImage, superiorClipImage;
  if (args_info.inferiorclipimage_given)
  {
    if (args_info.verbose_flag)
      std::cout << "Reading inferior clip image " << args_info.inferiorclipimage_arg << "..." << std::endl;
    // Read an existing image to initialize the attenuation map
    inferiorClipImage = itk::ReadImage<ClipImageType>(args_info.inferiorclipimage_arg);
  }
  if (args_info.superiorclipimage_given)
  {
    if (args_info.verbose_flag)
      std::cout << "Reading superior clip image " << args_info.superiorclipimage_arg << "..." << std::endl;
    // Read an existing image to initialize the attenuation map
    superiorClipImage = itk::ReadImage<ClipImageType>(args_info.superiorclipimage_arg);
  }

  // Create forward projection image filter
  if (args_info.verbose_flag)
    std::cout << "Projecting volume..." << std::endl;

  rtk::ForwardProjectionImageFilter<OutputImageType, OutputImageType>::Pointer forwardProjection;

  switch (args_info.fp_arg)
  {
    case (fp_arg_Joseph):
      forwardProjection = rtk::JosephForwardProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case (fp_arg_JosephAttenuated):
      forwardProjection = rtk::JosephForwardAttenuatedProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case (fp_arg_Zeng):
      forwardProjection = rtk::ZengForwardProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case (fp_arg_MIP):
      forwardProjection = rtk::MaximumIntensityProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case (fp_arg_CudaRayCast):
#ifdef RTK_USE_CUDA
      forwardProjection = rtk::CudaForwardProjectionImageFilter<OutputImageType, OutputImageType>::New();
      dynamic_cast<rtk::CudaForwardProjectionImageFilter<OutputImageType, OutputImageType> *>(
        forwardProjection.GetPointer())
        ->SetStepSize(args_info.step_arg);
#else
      std::cerr << "The program has not been compiled with cuda option" << std::endl;
      return EXIT_FAILURE;
#endif
      break;
    default:
      std::cerr << "Unhandled --method value." << std::endl;
      return EXIT_FAILURE;
  }
  forwardProjection->SetInput(constantImageSource->GetOutput());
  forwardProjection->SetInput(1, inputVolume);
  if (args_info.attenuationmap_given)
    forwardProjection->SetInput(2, attenuationMap);
  if (args_info.inferiorclipimage_given)
  {
    if (args_info.fp_arg == fp_arg_Joseph)
    {
      dynamic_cast<rtk::JosephForwardProjectionImageFilter<OutputImageType, OutputImageType> *>(
        forwardProjection.GetPointer())
        ->SetInferiorClipImage(inferiorClipImage);
    }
    else if (args_info.fp_arg == fp_arg_JosephAttenuated)
    {
      dynamic_cast<rtk::JosephForwardAttenuatedProjectionImageFilter<OutputImageType, OutputImageType> *>(
        forwardProjection.GetPointer())
        ->SetInferiorClipImage(inferiorClipImage);
    }
    else if (args_info.fp_arg == fp_arg_MIP)
    {
      dynamic_cast<rtk::MaximumIntensityProjectionImageFilter<OutputImageType, OutputImageType> *>(
        forwardProjection.GetPointer())
        ->SetInferiorClipImage(inferiorClipImage);
    }
  }
  if (args_info.superiorclipimage_given)
  {
    if (args_info.fp_arg == fp_arg_Joseph)
    {
      dynamic_cast<rtk::JosephForwardProjectionImageFilter<OutputImageType, OutputImageType> *>(
        forwardProjection.GetPointer())
        ->SetSuperiorClipImage(superiorClipImage);
    }
    else if (args_info.fp_arg == fp_arg_JosephAttenuated)
    {
      dynamic_cast<rtk::JosephForwardAttenuatedProjectionImageFilter<OutputImageType, OutputImageType> *>(
        forwardProjection.GetPointer())
        ->SetSuperiorClipImage(superiorClipImage);
    }
    else if (args_info.fp_arg == fp_arg_MIP)
    {
      dynamic_cast<rtk::MaximumIntensityProjectionImageFilter<OutputImageType, OutputImageType> *>(
        forwardProjection.GetPointer())
        ->SetSuperiorClipImage(superiorClipImage);
    }
  }
  if (args_info.sigmazero_given && args_info.fp_arg == fp_arg_Zeng)
    dynamic_cast<rtk::ZengForwardProjectionImageFilter<OutputImageType, OutputImageType> *>(
      forwardProjection.GetPointer())
      ->SetSigmaZero(args_info.sigmazero_arg);
  if (args_info.alphapsf_given && args_info.fp_arg == fp_arg_Zeng)
    dynamic_cast<rtk::ZengForwardProjectionImageFilter<OutputImageType, OutputImageType> *>(
      forwardProjection.GetPointer())
      ->SetAlpha(args_info.alphapsf_arg);
  forwardProjection->SetGeometry(geometry);
  if (!args_info.lowmem_flag)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(forwardProjection->Update())
  }

  // Write
  if (args_info.verbose_flag)
    std::cout << "Writing... " << std::endl;
  auto writer = itk::ImageFileWriter<OutputImageType>::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(forwardProjection->GetOutput());
  if (args_info.lowmem_flag)
  {
    writer->SetNumberOfStreamDivisions(sizeOutput[2]);
  }
  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  return EXIT_SUCCESS;
}
