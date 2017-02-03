/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtkvarianprobeamgeometry_ggo.h"
#include "rtkMacro.h"
#include "rtkVarianProBeamGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkGgoFunctions.h"

int main(int argc, char * argv[])
{
  GGO(rtkvarianprobeamgeometry, args_info);

  // Create geometry reader
  rtk::VarianProBeamGeometryReader::Pointer reader;
  reader = rtk::VarianProBeamGeometryReader::New();
  reader->SetXMLFileName(args_info.xml_file_arg);
  reader->SetProjectionsFileNames( rtk::GetProjectionsFileNamesFromGgo(args_info) );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->UpdateOutputData() )

  // Write
  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter =
    rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject( reader->GetGeometry() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( xmlWriter->WriteFile() )

  return EXIT_SUCCESS;
}
