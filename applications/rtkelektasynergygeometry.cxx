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

#include "rtkelektasynergygeometry_ggo.h"
#include "rtkDbf.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

#include <itkTimeProbe.h>

std::string GetImageIDFromDicomUID(const args_info_rtkelektasynergygeometry &args_info)
{
  // Open image database file
  rtk::DbfFile dbImage(args_info.image_db_arg);

  if (!dbImage.is_open() ) {
    std::cerr << "Couldn't open " << args_info.image_db_arg << std::endl;
    exit(EXIT_FAILURE);
    }

  // Search for correct record
  bool bReadOk;
  do {
    bReadOk = dbImage.ReadNextRecord();
    }
  while(bReadOk && std::string(args_info.dicom_uid_arg) != dbImage.GetFieldAsString("DICOM_UID") );

  // Error message if not found
  if(!bReadOk)
    {
    std::cerr << "Couldn't find acquisition with DICOM_UID " << args_info.dicom_uid_arg
              << " in table " << args_info.image_db_arg << std::endl;
    exit(EXIT_FAILURE);
    }

  return dbImage.GetFieldAsString("DBID");
}

void GetProjInfoFromDB(const std::string &imageID,
                       const args_info_rtkelektasynergygeometry &args_info,
                       std::vector<float> &projAngle,
                       std::vector<float> &projFlexX,
                       std::vector<float> &projFlexY)
{
  // Open frame database file
  rtk::DbfFile dbFrame(args_info.frame_db_arg);

  if (!dbFrame.is_open() ) {
    std::cerr << "Couldn't open " << args_info.frame_db_arg << std::endl;
    exit(EXIT_FAILURE);
    }

  // Go through the database, select correct records and get data
  while( dbFrame.ReadNextRecord() )
    {
    if(dbFrame.GetFieldAsString("IMA_DBID") == imageID)
      {
      projAngle.push_back(dbFrame.GetFieldAsDouble("PROJ_ANG") );
      projFlexX.push_back(dbFrame.GetFieldAsDouble("U_CENTRE") );
      projFlexY.push_back(dbFrame.GetFieldAsDouble("V_CENTRE") );
      }
    }
}

int main(int argc, char * argv[])
{
  GGO(rtkelektasynergygeometry, args_info);

  // RTK geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();

  // Get information from Synergy database
  std::vector<float> projAngle, projFlexX, projFlexY;
  GetProjInfoFromDB(GetImageIDFromDicomUID(args_info), args_info, projAngle, projFlexX, projFlexY);

  // Projection matrices
  for(unsigned int noProj=0; noProj<projAngle.size(); noProj++)
    {
    geometry->AddProjection(args_info.sid_arg,
                            args_info.sdd_arg,
                            projAngle[noProj],
                            -projFlexX[noProj],
                            -projFlexY[noProj]);
    }

  // Write
  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter =
    rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject(&(*geometry) );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( xmlWriter->WriteFile() )

  return EXIT_SUCCESS;
}
