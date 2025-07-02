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

#include "rtkimagxgeometry_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkImagXGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

int
main(int argc, char * argv[])
{
  GGO(rtkimagxgeometry, args_info);

  // Image Type
  constexpr unsigned int Dimension = 3;

  // Create geometry reader
  auto imagxReader = rtk::ImagXGeometryReader<itk::Image<float, Dimension>>::New();
  imagxReader->SetProjectionsFileNames(rtk::GetProjectionsFileNamesFromGgo(args_info));
  if (args_info.calibration_given)
  {
    imagxReader->SetCalibrationXMLFileName(args_info.calibration_arg);
  }
  if (args_info.room_setup_given)
  {
    imagxReader->SetRoomXMLFileName(args_info.room_setup_arg);
  }
  TRY_AND_EXIT_ON_ITK_EXCEPTION(imagxReader->UpdateOutputData())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(rtk::WriteGeometry(imagxReader->GetGeometry(), args_info.output_arg))

  return EXIT_SUCCESS;
}
