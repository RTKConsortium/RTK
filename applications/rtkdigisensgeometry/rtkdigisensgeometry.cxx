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

#include "rtkdigisensgeometry_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkDigisensGeometryReader.h"

int
main(int argc, char * argv[])
{
  GGO(rtkdigisensgeometry, args_info);

  // Create geometry reader
  auto reader = rtk::DigisensGeometryReader::New();
  reader->SetXMLFileName(args_info.xml_file_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->UpdateOutputData())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(rtk::WriteGeometry(reader->GetGeometry(), args_info.output_arg))

  return EXIT_SUCCESS;
}
