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

#include "rtkrayboxintersection_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"

#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkrayboxintersection, args_info);

  constexpr unsigned int Dimension = 3;

  using OutputImageType = itk::Image<float, Dimension>;

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

  // Create a stack of empty projection images
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkrayboxintersection>(constantImageSource,
                                                                                               args_info);

  // Adjust size according to geometry
  constantImageSource->SetSize(itk::MakeSize(
    constantImageSource->GetSize()[0], constantImageSource->GetSize()[1], geometry->GetGantryAngles().size()));

  // Input reader
  OutputImageType::Pointer input;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(input = itk::ReadImage<OutputImageType>(args_info.input_arg))

  // Create projection image filter
  auto rbi = rtk::RayBoxIntersectionImageFilter<OutputImageType, OutputImageType>::New();
  rbi->SetInput(constantImageSource->GetOutput());
  rbi->SetBoxFromImage(input);
  rbi->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(rbi->Update())

  // Write
  if (args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(rbi->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
