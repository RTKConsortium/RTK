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

#ifndef _rtkThreeDCircularProjectionGeometryXMLFileReader_cxx
#define _rtkThreeDCircularProjectionGeometryXMLFileReader_cxx

#include "rtkThreeDCircularProjectionGeometryXMLFileReader.h"

#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>
#include <itkIOCommon.h>

#include <iomanip>

namespace rtk
{

ThreeDCircularProjectionGeometryXMLFileReader::
ThreeDCircularProjectionGeometryXMLFileReader():
  m_Geometry(GeometryType::New() ),
  m_CurCharacterData(""),
  m_InPlaneAngle(0.),
  m_OutOfPlaneAngle(0.),
  m_GantryAngle(0.),
  m_SourceToIsocenterDistance(0.),
  m_SourceOffsetX(0.),
  m_SourceOffsetY(0.),
  m_SourceToDetectorDistance(0.),
  m_ProjectionOffsetX(0.),
  m_ProjectionOffsetY(0.),
  m_CollimationUInf(std::numeric_limits< double >::max()),
  m_CollimationUSup(std::numeric_limits< double >::max()),
  m_CollimationVInf(std::numeric_limits< double >::max()),
  m_CollimationVSup(std::numeric_limits< double >::max()),
  m_Version(0)
{
  this->m_OutputObject = &(*m_Geometry);
}

int
ThreeDCircularProjectionGeometryXMLFileReader::
CanReadFile(const char *name)
{
  if(!itksys::SystemTools::FileExists(name) ||
     itksys::SystemTools::FileIsDirectory(name) ||
     itksys::SystemTools::FileLength(name) == 0)
    return 0;
  return 1;
}

void
ThreeDCircularProjectionGeometryXMLFileReader::
StartElement(const char * name,const char **atts)
{
  m_CurCharacterData = "";
  this->StartElement(name);

  // Check on last version of file format. Warning if not.
  if( std::string(name) == "RTKThreeDCircularGeometry" )
    {
    while( (*atts) != ITK_NULLPTR )
      {
      if( std::string(atts[0]) == "version" )
        m_Version = atoi(atts[1]);
      atts += 2;
      }
    // Version 3 is backward compatible with version 2
    if(  m_Version != this->CurrentVersion &&
       !(m_Version == 2 && this->CurrentVersion == 3) )
      itkGenericExceptionMacro(<< "Incompatible version of input geometry (v" << m_Version
                               << ") with current geometry (v" << this->CurrentVersion
                               << "). You must re-generate your geometry file again.");
    this->m_OutputObject->Clear();
    }
}

void
ThreeDCircularProjectionGeometryXMLFileReader::
StartElement(const char * itkNotUsed(name))
{
}

void
ThreeDCircularProjectionGeometryXMLFileReader::
EndElement(const char *name)
{
  if(itksys::SystemTools::Strucmp(name, "InPlaneAngle") == 0)
    m_InPlaneAngle = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "GantryAngle") == 0 ||
     itksys::SystemTools::Strucmp(name, "Angle") == 0) // Second one for backward compatibility
    m_GantryAngle = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "OutOfPlaneAngle") == 0)
    m_OutOfPlaneAngle = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "SourceToIsocenterDistance") == 0)
    m_SourceToIsocenterDistance = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "SourceOffsetX") == 0)
    m_SourceOffsetX = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "SourceOffsetY") == 0)
    m_SourceOffsetY = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "SourceToDetectorDistance") == 0)
    m_SourceToDetectorDistance = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "ProjectionOffsetX") == 0)
    m_ProjectionOffsetX = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "ProjectionOffsetY") == 0)
    m_ProjectionOffsetY = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "RadiusCylindricalDetector") == 0)
    {
    double radiusCylindricalDetector = atof(this->m_CurCharacterData.c_str() );
    this->m_OutputObject->SetRadiusCylindricalDetector(radiusCylindricalDetector);
    }

  if(itksys::SystemTools::Strucmp(name, "CollimationUInf") == 0)
    m_CollimationUInf = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "CollimationUSup") == 0)
    m_CollimationUSup = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "CollimationVInf") == 0)
    m_CollimationVInf = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "CollimationVSup") == 0)
    m_CollimationVSup = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "Matrix") == 0)
    {
    std::istringstream iss(this->m_CurCharacterData);
    double value = 0.;
    for(unsigned int i=0; i<m_Matrix.RowDimensions; i++)
      for(unsigned int j=0; j<m_Matrix.ColumnDimensions; j++)
        {
        iss >> value;
        m_Matrix[i][j] = value;
        }
    }

  if(itksys::SystemTools::Strucmp(name, "Projection") == 0)
    {
    this->m_OutputObject->AddProjection(m_SourceToIsocenterDistance,
                                        m_SourceToDetectorDistance,
                                        m_GantryAngle,
                                        m_ProjectionOffsetX,
                                        m_ProjectionOffsetY,
                                        m_OutOfPlaneAngle,
                                        m_InPlaneAngle,
                                        m_SourceOffsetX,
                                        m_SourceOffsetY);

    this->m_OutputObject->SetCollimationOfLastProjection(m_CollimationUInf,
                                                         m_CollimationUSup,
                                                         m_CollimationVInf,
                                                         m_CollimationVSup);

    for(unsigned int i=0; i<m_Matrix.RowDimensions; i++)
      for(unsigned int j=0; j<m_Matrix.ColumnDimensions; j++)
        {
        // Tolerance can not be vcl_numeric_limits<double>::epsilon(), too strict
        // 0.001 is a random choice to catch "large" inconsistencies
        if( fabs(m_Matrix[i][j]-m_OutputObject->GetMatrices().back()[i][j]) > 0.001 )
          {
          itkGenericExceptionMacro(<< "Matrix and parameters are not consistent."
                                   << std::endl << "Read matrix from geometry file: " 
                                   << std::endl << m_Matrix
                                   << std::endl << "Computed matrix from parameters:"
                                   << std::endl << m_OutputObject->GetMatrices().back());
          }
        }
    }
}

void
ThreeDCircularProjectionGeometryXMLFileReader::
CharacterDataHandler(const char *inData, int inLength)
{
  for(int i = 0; i < inLength; i++)
    m_CurCharacterData = m_CurCharacterData + inData[i];
}

}

#endif
