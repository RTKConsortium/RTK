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


#include "rtkElektaSynergyGeometryReader.h"
#include "rtkDbf.h"
#include "rtkMacro.h"

rtk::ElektaSynergyGeometryReader
::ElektaSynergyGeometryReader():
  m_Geometry(ITK_NULLPTR),
  m_DicomUID(""),
  m_ImageDbfFileName("IMAGE.DBF"),
  m_FrameDbfFileName("FRAME.DBF")
{
}

std::string
rtk::ElektaSynergyGeometryReader
::GetImageIDFromDicomUID()
{
  // Open frame database file
  rtk::DbfFile dbImage(m_ImageDbfFileName);
  if( !dbImage.is_open() )
    itkGenericExceptionMacro( << "Couldn't open " 
                              << m_ImageDbfFileName);

  // Search for correct record
  bool bReadOk;
  do {
    bReadOk = dbImage.ReadNextRecord();
    }
  while(bReadOk && std::string(m_DicomUID) != dbImage.GetFieldAsString("DICOM_UID") );

  // Error message if not found
  if(!bReadOk)
    {
    itkGenericExceptionMacro( << "Couldn't find acquisition with DICOM_UID "
                              << m_DicomUID
                              << " in table "
                              << m_ImageDbfFileName );
    }

  return dbImage.GetFieldAsString("DBID");
}

void
rtk::ElektaSynergyGeometryReader
::GetProjInfoFromDB(const std::string &imageID,
                    std::vector<float> &projAngle,
                    std::vector<float> &projFlexX,
                    std::vector<float> &projFlexY)
{
  // Open frame database file
  rtk::DbfFile dbFrame(m_FrameDbfFileName);
  if( !dbFrame.is_open() )
    itkGenericExceptionMacro( << "Couldn't open " 
                              << m_FrameDbfFileName);

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

void
rtk::ElektaSynergyGeometryReader
::GenerateData()
{
  // Create new RTK geometry object
  m_Geometry = GeometryType::New();

  // Get information from Synergy database
  std::vector<float> projAngle, projFlexX, projFlexY;
  GetProjInfoFromDB( GetImageIDFromDicomUID(), projAngle, projFlexX, projFlexY);

  // Projection matrices
  for(unsigned int noProj=0; noProj<projAngle.size(); noProj++)
    {
    m_Geometry->AddProjection(1000.,
                            1536.,
                            projAngle[noProj],
                            -projFlexX[noProj],
                            -projFlexY[noProj]);
    }
}
