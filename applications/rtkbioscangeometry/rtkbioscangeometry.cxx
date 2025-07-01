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

#include "rtkbioscangeometry_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkBioscanGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

int
main(int argc, char * argv[])
{
  GGO(rtkbioscangeometry, args_info);

  // Create geometry reader
  auto bioscanReader = rtk::BioscanGeometryReader::New();
  bioscanReader->SetProjectionsFileNames(rtk::GetProjectionsFileNamesFromGgo(args_info));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(bioscanReader->UpdateOutputData())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(rtk::WriteGeometry(
    const_cast<rtk::ThreeDCircularProjectionGeometry *>(bioscanReader->GetGeometry()), args_info.output_arg))

  return EXIT_SUCCESS;
}
