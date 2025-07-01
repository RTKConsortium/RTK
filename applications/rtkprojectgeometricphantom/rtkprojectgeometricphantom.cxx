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

#include "rtkprojectgeometricphantom_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkProjectGeometricPhantomImageFilter.h"

#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkprojectgeometricphantom, args_info);

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
  auto constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkprojectgeometricphantom>(constantImageSource,
                                                                                                    args_info);

  // Adjust size according to geometry
  constantImageSource->SetSize(itk::MakeSize(
    constantImageSource->GetSize()[0], constantImageSource->GetSize()[1], geometry->GetGantryAngles().size()));

  using PPCType = rtk::ProjectGeometricPhantomImageFilter<OutputImageType, OutputImageType>;

  // Offset, scale, rotation
  PPCType::VectorType offset(0.);
  if (args_info.offset_given)
  {
    if (args_info.offset_given > 3)
    {
      std::cerr << "--offset needs up to 3 values" << std::endl;
      exit(EXIT_FAILURE);
    }
    offset[0] = args_info.offset_arg[0];
    offset[1] = args_info.offset_arg[1];
    offset[2] = args_info.offset_arg[2];
  }
  PPCType::VectorType scale;
  scale.Fill(args_info.phantomscale_arg[0]);
  if (args_info.phantomscale_given)
  {
    if (args_info.phantomscale_given > 3)
    {
      std::cerr << "--phantomscale needs up to 3 values" << std::endl;
      exit(EXIT_FAILURE);
    }
    for (unsigned int i = 0; i < std::min(args_info.phantomscale_given, Dimension); i++)
      scale[i] = args_info.phantomscale_arg[i];
  }
  PPCType::RotationMatrixType rot;
  rot.SetIdentity();
  if (args_info.rotation_given)
  {
    if (args_info.rotation_given != 9)
    {
      std::cerr << "--rotation needs exactly 9 values" << std::endl;
      exit(EXIT_FAILURE);
    }
    for (unsigned int i = 0; i < Dimension; i++)
      for (unsigned int j = 0; j < Dimension; j++)
        rot[i][j] = args_info.rotation_arg[i * Dimension + j];
  }

  auto ppc = PPCType::New();
  ppc->SetInput(constantImageSource->GetOutput());
  ppc->SetGeometry(geometry);
  ppc->SetPhantomScale(scale);
  ppc->SetOriginOffset(offset);
  ppc->SetRotationMatrix(rot);
  ppc->SetConfigFile(args_info.phantomfile_arg);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(ppc->Update())

  // Write
  if (args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(ppc->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
