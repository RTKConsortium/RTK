#ifndef _rtkThreeDCircularGeometryXMLFile_cxx
#define _rtkThreeDCircularGeometryXMLFile_cxx

#include "rtkThreeDCircularGeometryXMLFile.h"

#include <itkXMLFile.h>
#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>
#include <itkIOCommon.h>

#include <iomanip>

namespace rtk
{

void
ThreeDCircularGeometryXMLFileReader::
StartElement(const char * name)
{
  if(itksys::SystemTools::Strucmp(name, "Projection") == 0)
  {
    m_RotationAngle = 0.0;
    m_ProjectionOffsetX = 0.0;
    m_ProjectionOffsetY = 0.0;
  }
}

void
ThreeDCircularGeometryXMLFileReader::
EndElement(const char *name)
{
  if(itksys::SystemTools::Strucmp(name, "SourceToDetectorDistance") == 0)
    this->m_OutputObject->SetSourceToDetectorDistance(atof(this->m_CurCharacterData.c_str()));

  if(itksys::SystemTools::Strucmp(name, "SourceToIsocenterDistance") == 0)
    this->m_OutputObject->SetSourceToIsocenterDistance(atof(this->m_CurCharacterData.c_str()));

  if(itksys::SystemTools::Strucmp(name, "Angle") == 0)
    m_RotationAngle = atof(this->m_CurCharacterData.c_str());

  if(itksys::SystemTools::Strucmp(name, "ProjectionOffsetX") == 0)
    m_ProjectionOffsetX = atof(this->m_CurCharacterData.c_str());

  if(itksys::SystemTools::Strucmp(name, "ProjectionOffsetY") == 0)
    m_ProjectionOffsetY = atof(this->m_CurCharacterData.c_str());

  if(itksys::SystemTools::Strucmp(name, "Projection") == 0)
    this->m_OutputObject->AddProjection(m_RotationAngle, m_ProjectionOffsetX, m_ProjectionOffsetY);
}

int
ThreeDCircularGeometryXMLFileWriter::
WriteFile()
{
  std::ofstream output(this->m_Filename.c_str());
  const int maxDigits = 15;
  output.precision(maxDigits);
  std::string indent("  ");

  this->WriteStartElement("?xml version=\"1.0\"?",output);
  output << std::endl;
  this->WriteStartElement("!DOCTYPE RTKGEOMETRY",output);
  output << std::endl;
  this->WriteStartElement("RTKThreeDCircularGeometry",output);
  output << std::endl;

  output << indent;
  this->WriteStartElement("SourceToDetectorDistance",output);
  output << this->m_InputObject->GetSourceToDetectorDistance();
  this->WriteEndElement("SourceToDetectorDistance",output);
  output << std::endl;

  output << indent;
  this->WriteStartElement("SourceToIsocenterDistance",output);
  output << this->m_InputObject->GetSourceToIsocenterDistance();
  this->WriteEndElement("SourceToIsocenterDistance",output);
  output << std::endl;

  for(unsigned int i = 0; i<this->m_InputObject->GetMatrices().size(); i++)
    {
    output << indent;
    this->WriteStartElement("Projection",output);
    output << std::endl;

    //Angle
    output << indent << indent;
    this->WriteStartElement("Angle",output);
    output << this->m_InputObject->GetRotationAngles()[i];
    this->WriteEndElement("Angle",output);
    output << std::endl;

    //Projection offset
    output << indent << indent;
    this->WriteStartElement("ProjectionOffsetX",output);
    output << this->m_InputObject->GetProjectionOffsetsX()[i];
    this->WriteEndElement("ProjectionOffsetX",output);
    output << std::endl;
    output << indent << indent;
    this->WriteStartElement("ProjectionOffsetY",output);
    output << this->m_InputObject->GetProjectionOffsetsX()[i];
    this->WriteEndElement("ProjectionOffsetY",output);
    output << std::endl;

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

}

#endif
