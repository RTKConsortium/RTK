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

#ifndef _rtkThreeDCircularProjectionGeometryXMLFileWriter_cxx
#define _rtkThreeDCircularProjectionGeometryXMLFileWriter_cxx

#include "rtkThreeDCircularProjectionGeometryXMLFileReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFileWriter.h"

#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>
#include <itkIOCommon.h>

#include <iomanip>

namespace rtk
{

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
