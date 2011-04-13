#ifndef _itkThreeDCircularProjectionGeometryXMLFile_cxx
#define _itkThreeDCircularProjectionGeometryXMLFile_cxx

#include "itkThreeDCircularProjectionGeometryXMLFile.h"

#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>
#include <itkIOCommon.h>

#include <iomanip>

namespace itk
{

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
}

void
ThreeDCircularProjectionGeometryXMLFileReader::
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
ThreeDCircularProjectionGeometryXMLFileReader::
EndElement(const char *name)
{
  if(itksys::SystemTools::Strucmp(name, "SourceToDetectorDistance") == 0)
    this->m_OutputObject->SetSourceToDetectorDistance(atof(this->m_CurCharacterData.c_str() ) );

  if(itksys::SystemTools::Strucmp(name, "SourceToIsocenterDistance") == 0)
    this->m_OutputObject->SetSourceToIsocenterDistance(atof(this->m_CurCharacterData.c_str() ) );

  if(itksys::SystemTools::Strucmp(name, "ProjectionScalingX") == 0)
    this->m_OutputObject->SetProjectionScalingX(atof(this->m_CurCharacterData.c_str() ) );

  if(itksys::SystemTools::Strucmp(name, "ProjectionScalingY") == 0)
    this->m_OutputObject->SetProjectionScalingY(atof(this->m_CurCharacterData.c_str() ) );

  if(itksys::SystemTools::Strucmp(name, "RotationCenter") == 0)
    {
    ThreeDCircularProjectionGeometry::VectorType vec;
    std::istringstream                           iss(m_CurCharacterData);
    iss >> vec;
    this->m_OutputObject->SetRotationCenter(vec);
    }

  if(itksys::SystemTools::Strucmp(name, "RotationAxis") == 0)
    {
    ThreeDCircularProjectionGeometry::VectorType vec;
    std::istringstream                           iss(m_CurCharacterData);
    iss >> vec;
    this->m_OutputObject->SetRotationAxis(vec);
    }

  if(itksys::SystemTools::Strucmp(name, "Angle") == 0)
    m_RotationAngle = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "ProjectionOffsetX") == 0)
    m_ProjectionOffsetX = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "ProjectionOffsetY") == 0)
    m_ProjectionOffsetY = atof(this->m_CurCharacterData.c_str() );

  if(itksys::SystemTools::Strucmp(name, "Projection") == 0)
    this->m_OutputObject->AddProjection(m_RotationAngle, m_ProjectionOffsetX, m_ProjectionOffsetY);
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
  std::ofstream output(this->m_Filename.c_str() );
  const int     maxDigits = 15;

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

  if(this->m_InputObject->GetProjectionScalingX() != 1.)
    {
    output << indent;
    this->WriteStartElement("ProjectionScalingX",output);
    output << this->m_InputObject->GetProjectionScalingX();
    this->WriteEndElement("ProjectionScalingX",output);
    output << std::endl;
    }

  if(this->m_InputObject->GetProjectionScalingY() != 1.)
    {
    output << indent;
    this->WriteStartElement("ProjectionScalingY",output);
    output << this->m_InputObject->GetProjectionScalingX();
    this->WriteEndElement("ProjectionScalingY",output);
    output << std::endl;
    }

  output << indent;
  this->WriteStartElement("RotationCenter",output);
  output << this->m_InputObject->GetRotationCenter()[0] << ' '
         << this->m_InputObject->GetRotationCenter()[1] << ' '
         << this->m_InputObject->GetRotationCenter()[2];
  this->WriteEndElement("RotationCenter",output);
  output << std::endl;

  output << indent;
  this->WriteStartElement("RotationAxis",output);
  output << this->m_InputObject->GetRotationAxis()[0] << ' '
         << this->m_InputObject->GetRotationAxis()[1] << ' '
         << this->m_InputObject->GetRotationAxis()[2];
  this->WriteEndElement("RotationAxis",output);
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
    output << this->m_InputObject->GetProjectionOffsetsY()[i];
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
