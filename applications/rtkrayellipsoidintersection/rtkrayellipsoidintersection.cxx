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

#include "rtkrayellipsoidintersection_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkrayellipsoidintersection, args_info);

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
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkrayellipsoidintersection>(
    constantImageSource, args_info);

  // Adjust size according to geometry
  constantImageSource->SetSize(itk::MakeSize(
    constantImageSource->GetSize()[0], constantImageSource->GetSize()[1], geometry->GetGantryAngles().size()));

  // Create projection image filter
  auto rei = rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType>::New();

  rei->SetDensity(args_info.mult_arg);
  if (args_info.axes_given > 0)
    rei->SetAxis[0] = args_info.axes_arg[0];
  if (args_info.axes_given > 1)
    rei->SetAxis[1] = args_info.axes_arg[1];
  if (args_info.axes_given > 2)
    rei->SetAxis[2] = args_info.axes_arg[2];
  if (args_info.center_given > 0)
    rei->SetCenter[0] = args_info.center_arg[0];
  if (args_info.center_given > 1)
    rei->SetCenter[1] = args_info.center_arg[1];
  if (args_info.center_given > 2)
    rei->SetCenter[2] = args_info.center_arg[2];
  if (args_info.rotation_given > 0)
  {
    rei->SetRotate(true);
    rei->SetAngle(args_info.rotation_arg[0]);
  }

  rei->SetInput(constantImageSource->GetOutput());
  rei->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(rei->Update())

  // Write
  if (args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(rei->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
