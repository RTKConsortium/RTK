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

#include "rtkorageometry_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkOraGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

int
main(int argc, char * argv[])
{
  GGO(rtkorageometry, args_info);

  rtk::OraGeometryReader::MarginVectorType margin;
  margin.Fill(args_info.margin_arg[0]);
  for (unsigned int i = 0; i < std::min(args_info.margin_given, margin.GetVectorDimension()); i++)
    margin[i] = args_info.margin_arg[i];

  // Create geometry reader
  rtk::OraGeometryReader::Pointer oraReader = rtk::OraGeometryReader::New();
  oraReader->SetProjectionsFileNames(rtk::GetProjectionsFileNamesFromGgo(args_info));
  oraReader->SetCollimationMargin(margin);
  oraReader->SetOptiTrackObjectID(args_info.optitrack_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(oraReader->UpdateOutputData())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(rtk::WriteGeometry(
    const_cast<rtk::ThreeDCircularProjectionGeometry *>(oraReader->GetGeometry()), args_info.output_arg))

  return EXIT_SUCCESS;
}
