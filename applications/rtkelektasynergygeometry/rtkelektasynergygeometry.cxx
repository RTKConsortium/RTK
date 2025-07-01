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

#include "rtkelektasynergygeometry_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkElektaSynergyGeometryReader.h"
#include "rtkElektaXVI5GeometryXMLFileReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

int
main(int argc, char * argv[])
{
  GGO(rtkelektasynergygeometry, args_info);

  if (args_info.image_db_given && args_info.frame_db_given && args_info.dicom_uid_given && !args_info.xml_given)
  {
    // Create geometry reader
    auto reader = rtk::ElektaSynergyGeometryReader::New();
    reader->SetDicomUID(args_info.dicom_uid_arg);
    reader->SetImageDbfFileName(args_info.image_db_arg);
    reader->SetFrameDbfFileName(args_info.frame_db_arg);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->UpdateOutputData())

    // Write
    TRY_AND_EXIT_ON_ITK_EXCEPTION(rtk::WriteGeometry(reader->GetGeometry(), args_info.output_arg))


    return EXIT_SUCCESS;
  }
  else if (!args_info.image_db_given && !args_info.frame_db_given && !args_info.dicom_uid_given && args_info.xml_given)
  {
    // Create geometry reader
    rtk::ElektaXVI5GeometryXMLFileReader::Pointer geometryReader;
    geometryReader = rtk::ElektaXVI5GeometryXMLFileReader::New();
    geometryReader->SetFilename(args_info.xml_arg);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(geometryReader->GenerateOutputInformation());

    // Write
    TRY_AND_EXIT_ON_ITK_EXCEPTION(rtk::WriteGeometry(geometryReader->GetGeometry(), args_info.output_arg))

    return EXIT_SUCCESS;
  }

  std::cerr << "You must either provide image_db, frame_db and dicom_uid"
            << "for versions up to v4 or xml starting with v5." << std::endl;

  return EXIT_FAILURE;
}
