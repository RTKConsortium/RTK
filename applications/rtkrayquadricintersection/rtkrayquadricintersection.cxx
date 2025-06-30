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

#include "rtkrayquadricintersection_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayQuadricIntersectionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkrayquadricintersection, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

  // Create a stack of empty projection images
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkrayquadricintersection>(constantImageSource,
                                                                                                   args_info);

  // Adjust size according to geometry
  constantImageSource->SetSize(itk::MakeSize(
    constantImageSource->GetSize()[0], constantImageSource->GetSize()[1], geometry->GetGantryAngles().size()));

  // Create projection image filter
  using RQIType = rtk::RayQuadricIntersectionImageFilter<OutputImageType, OutputImageType>;
  RQIType::Pointer rqi = RQIType::New();
  rqi->SetInput(constantImageSource->GetOutput());
  if (args_info.parameters_given > 0)
    rqi->SetA(args_info.parameters_arg[0]);
  if (args_info.parameters_given > 1)
    rqi->SetB(args_info.parameters_arg[1]);
  if (args_info.parameters_given > 2)
    rqi->SetC(args_info.parameters_arg[2]);
  if (args_info.parameters_given > 3)
    rqi->SetD(args_info.parameters_arg[3]);
  if (args_info.parameters_given > 4)
    rqi->SetE(args_info.parameters_arg[4]);
  if (args_info.parameters_given > 5)
    rqi->SetF(args_info.parameters_arg[5]);
  if (args_info.parameters_given > 6)
    rqi->SetG(args_info.parameters_arg[6]);
  if (args_info.parameters_given > 7)
    rqi->SetH(args_info.parameters_arg[7]);
  if (args_info.parameters_given > 8)
    rqi->SetI(args_info.parameters_arg[8]);
  if (args_info.parameters_given > 9)
    rqi->SetJ(args_info.parameters_arg[9]);
  rqi->SetDensity(args_info.mult_arg);
  rqi->SetGeometry(geometry);
  if (args_info.planes_given)
  {
    if (args_info.planes_given % 4 != 0)
    {
      std::cerr << "--plane requires four parameters" << std::endl;
      exit(EXIT_FAILURE);
    }
    for (unsigned int i = 0; i < args_info.planes_given / 4; i++)
    {
      RQIType::VectorType planeDir(args_info.planes_arg + i * 4);
      rqi->AddClipPlane(planeDir, args_info.planes_arg[i * 4 + 3]);
    }
  }
  TRY_AND_EXIT_ON_ITK_EXCEPTION(rqi->Update())

  // Write
  if (args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(rqi->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
