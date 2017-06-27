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

#ifndef _rtkThreeDCircularProjectionGeometryXMLFile_cxx
#define _rtkThreeDCircularProjectionGeometryXMLFile_cxx

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

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

int
ThreeDCircularProjectionGeometryXMLFileWriter::
CanWriteFile(const char * name)
{
  std::ofstream output(name);

  if(output.fail() )
    return false;
  return true;
}

int
ThreeDCircularProjectionGeometryXMLFileWriter::
WriteFile()
{
  if(this->m_InputObject->GetGantryAngles().size() == 0)
    itkGenericExceptionMacro(<< "Geometry object is empty, cannot write it");

  std::ofstream output(this->m_Filename.c_str() );
  const int     maxDigits = 15;

  output.precision(maxDigits);
  std::string indent("  ");

  this->WriteStartElement("?xml version=\"1.0\"?",output);
  output << std::endl;
  this->WriteStartElement("!DOCTYPE RTKGEOMETRY",output);
  output << std::endl;
  std::ostringstream startWithVersion;
  startWithVersion << "RTKThreeDCircularGeometry version=\""
                   << ThreeDCircularProjectionGeometryXMLFileReader::CurrentVersion
                   << '"';
  this->WriteStartElement(startWithVersion.str().c_str(),output);
  output << std::endl;
  
  // First, we test for each of the 9 parameters per projection if it's constant
  // over all projection images except GantryAngle which is supposed to be different
  // for all projections. If 0. for OutOfPlaneAngle, InPlaneAngle, projection and source
  // offsets X and Y, it is not written (default value).
  bool bSIDGlobal =
          WriteGlobalParameter(output, indent,
                               this->m_InputObject->GetSourceToIsocenterDistances(),
                               "SourceToIsocenterDistance");
  bool bSDDGlobal =
          WriteGlobalParameter(output, indent,
                               this->m_InputObject->GetSourceToDetectorDistances(),
                               "SourceToDetectorDistance");
  bool bSourceXGlobal =
          WriteGlobalParameter(output, indent,
                               this->m_InputObject->GetSourceOffsetsX(),
                               "SourceOffsetX");
  bool bSourceYGlobal =
          WriteGlobalParameter(output, indent,
                               this->m_InputObject->GetSourceOffsetsY(),
                               "SourceOffsetY");
  bool bProjXGlobal =
          WriteGlobalParameter(output, indent,
                               this->m_InputObject->GetProjectionOffsetsX(),
                               "ProjectionOffsetX");
  bool bProjYGlobal =
          WriteGlobalParameter(output, indent,
                               this->m_InputObject->GetProjectionOffsetsY(),
                               "ProjectionOffsetY");
  bool bInPlaneGlobal =
          WriteGlobalParameter(output, indent,
                               this->m_InputObject->GetInPlaneAngles(),
                               "InPlaneAngle",
                               true);
  bool bOutOfPlaneGlobal =
          WriteGlobalParameter(output, indent,
                               this->m_InputObject->GetOutOfPlaneAngles(),
                               "OutOfPlaneAngle",
                               true);

  bool bCollimationUInf =
          WriteGlobalParameter(output, indent,
                               this->m_InputObject->GetCollimationUInf(),
                               "CollimationUInf",
                               false,
                               std::numeric_limits<double>::max());

  bool bCollimationUSup =
          WriteGlobalParameter(output, indent,
                               this->m_InputObject->GetCollimationUSup(),
                               "CollimationUSup",
                               false,
                               std::numeric_limits<double>::max());

  bool bCollimationVInf =
          WriteGlobalParameter(output, indent,
                               this->m_InputObject->GetCollimationVInf(),
                               "CollimationVInf",
                               false,
                               std::numeric_limits<double>::max());

  bool bCollimationVSup =
          WriteGlobalParameter(output, indent,
                               this->m_InputObject->GetCollimationVSup(),
                               "CollimationVSup",
                               false,
                               std::numeric_limits<double>::max());

  const double radius = this->m_InputObject->GetRadiusCylindricalDetector();
  if (0. != radius)
    WriteLocalParameter(output, indent, radius, "RadiusCylindricalDetector");

  // Second, write per projection parameters (if corresponding parameter is not global)
  const double radiansToDegrees = 45. / vcl_atan(1.);
  for(unsigned int i = 0; i<this->m_InputObject->GetMatrices().size(); i++)
    {
    output << indent;
    this->WriteStartElement("Projection",output);
    output << std::endl;

    // Only the GantryAngle is necessarily projection specific
    WriteLocalParameter(output, indent,
                        radiansToDegrees * this->m_InputObject->GetGantryAngles()[i],
                        "GantryAngle");
    if(!bSIDGlobal)
      WriteLocalParameter(output, indent,
                          this->m_InputObject->GetSourceToIsocenterDistances()[i],
                          "SourceToIsocenterDistance");
    if(!bSDDGlobal)
      WriteLocalParameter(output, indent,
                          this->m_InputObject->GetSourceToDetectorDistances()[i],
                          "SourceToDetectorDistance");
    if(!bSourceXGlobal)
      WriteLocalParameter(output, indent,
                          this->m_InputObject->GetSourceOffsetsX()[i],
                          "SourceOffsetX");
    if(!bSourceYGlobal)
      WriteLocalParameter(output, indent,
                          this->m_InputObject->GetSourceOffsetsY()[i],
                          "SourceOffsetY");
    if(!bProjXGlobal)
      WriteLocalParameter(output, indent,
                          this->m_InputObject->GetProjectionOffsetsX()[i],
                          "ProjectionOffsetX");
    if(!bProjYGlobal)
      WriteLocalParameter(output, indent,
                          this->m_InputObject->GetProjectionOffsetsY()[i],
                          "ProjectionOffsetY");
    if(!bInPlaneGlobal)
      WriteLocalParameter(output, indent,
                          radiansToDegrees * this->m_InputObject->GetInPlaneAngles()[i],
                          "InPlaneAngle");
    if(!bOutOfPlaneGlobal)
      WriteLocalParameter(output, indent,
                          radiansToDegrees * this->m_InputObject->GetOutOfPlaneAngles()[i],
                          "OutOfPlaneAngle");

    if(!bCollimationUInf)
      WriteLocalParameter(output, indent,
                          this->m_InputObject->GetCollimationUInf()[i],
                          "CollimationUInf");

    if(!bCollimationUSup)
      WriteLocalParameter(output, indent,
                          this->m_InputObject->GetCollimationUSup()[i],
                          "CollimationUSup");

    if(!bCollimationVInf)
      WriteLocalParameter(output, indent,
                          this->m_InputObject->GetCollimationVInf()[i],
                          "CollimationVInf");

    if(!bCollimationVSup)
      WriteLocalParameter(output, indent,
                          this->m_InputObject->GetCollimationVSup()[i],
                          "CollimationVSup");

    //Matrix
    output << indent << indent;
    this->WriteStartElement("Matrix",output);
    output << std::endl;
    for(unsigned int j=0; j<3; j++)
      {
      output << indent << indent << indent;
      for(unsigned int k=0; k<4; k++)
        output << std::setw(maxDigits+4)
               << this->m_InputObject->GetMatrices()[i][j][k]
               << ' ';
      output.seekp(-1, std::ios_base::cur);
      output<< std::endl;
      }
    output << indent << indent;
    this->WriteEndElement("Matrix",output);
    output << std::endl;

    output << indent;
    this->WriteEndElement("Projection",output);
    output << std::endl;
    }

  this->WriteEndElement("RTKThreeDCircularGeometry",output);
  output << std::endl;

  return 0;
}

bool
ThreeDCircularProjectionGeometryXMLFileWriter::
WriteGlobalParameter(std::ofstream &output,
                     const std::string &indent,
                     const std::vector<double> &v,
                     const std::string &s,
                     bool convertToDegrees,
                     double defval)
{
  // Test if all values in vector v are equal. Return false if not.
  for(size_t i=0; i<v.size(); i++)
    if(v[i] != v[0])
      return false;

  // Write value in file if not 0.
  if (defval != v[0])
    {
    double val = v[0];
    if(convertToDegrees)
      val *= 45. / vcl_atan(1.);

    WriteLocalParameter(output, indent, val, s);
    }
  return true;
}

void
ThreeDCircularProjectionGeometryXMLFileWriter::
WriteLocalParameter(std::ofstream &output,
                    const std::string &indent,
                    const double &v,
                    const std::string &s)
{
  std::string ss(s);
  output << indent << indent;
  this->WriteStartElement(ss, output);
  output << v;
  this->WriteEndElement(ss, output);
  output << std::endl;
}

}

#endif
