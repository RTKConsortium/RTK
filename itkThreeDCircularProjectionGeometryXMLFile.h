#ifndef __itkThreeDCircularProjectionGeometryXMLFile_h
#define __itkThreeDCircularProjectionGeometryXMLFile_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include <itkXMLFile.h>
#include "itkThreeDCircularProjectionGeometry.h"

namespace itk
{

/** \class ThreeDCircularProjectionGeometryXMLFileReader
 *
 * Reads an XML-format file containing geometry for reconstruction
 */
class ThreeDCircularProjectionGeometryXMLFileReader :
  public XMLReader< ThreeDCircularProjectionGeometry >
{
public:
  /** Standard typedefs */
  typedef ThreeDCircularProjectionGeometryXMLFileReader Self;
  typedef XMLReader< ThreeDCircularProjectionGeometry > Superclass;
  typedef itk::SmartPointer<Self>                       Pointer;

  /** Convenient typedefs */
  typedef ThreeDCircularProjectionGeometry GeometryType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ThreeDCircularProjectionGeometryXMLFileReader, XMLFileReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Determine if a file can be read */
  int CanReadFile(const char* name);

protected:
  ThreeDCircularProjectionGeometryXMLFileReader() :  m_Geometry(GeometryType::New() ) {
    this->m_OutputObject = &(*m_Geometry);
  };
  ~ThreeDCircularProjectionGeometryXMLFileReader() {
  };

  /** Callback function -- called from XML parser with start-of-element
   * information.
   */
  void StartElement(const char * name,const char **atts);

  void StartElement(const char * name);

  void EndElement(const char *name);

  void CharacterDataHandler(const char *inData, int inLength);

private:
   //purposely not implemented
  ThreeDCircularProjectionGeometryXMLFileReader(const Self&);
  void operator=(const Self&);

  GeometryType::Pointer m_Geometry;

  std::string m_CurCharacterData;
  double      m_RotationAngle, m_ProjectionOffsetX, m_ProjectionOffsetY;
};

/** \class ThreeDCircularProjectionGeometryXMLFileWriter
 *
 * Writes an XML-format file containing geometry for reconstruction
 */
class ThreeDCircularProjectionGeometryXMLFileWriter :
  public XMLWriterBase< ThreeDCircularProjectionGeometry >
{
public:
  /** standard typedefs */
  typedef XMLWriterBase< ThreeDCircularProjectionGeometry > Superclass;
  typedef ThreeDCircularProjectionGeometryXMLFileWriter     Self;
  typedef itk::SmartPointer<Self>                           Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ThreeDCircularProjectionGeometryXMLFileWriter, XMLFileWriter);

  /** Test whether a file is writable. */
  int CanWriteFile(const char* name);

  /** Actually write out the file in question */
  int WriteFile();

protected:
  ThreeDCircularProjectionGeometryXMLFileWriter() {};
  ~ThreeDCircularProjectionGeometryXMLFileWriter() {};

private:
   //purposely not implemented
  ThreeDCircularProjectionGeometryXMLFileWriter(const Self&);
  void operator=(const Self&);

};
}

#endif
