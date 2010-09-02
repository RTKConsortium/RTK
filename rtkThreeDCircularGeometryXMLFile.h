#ifndef __rtkThreeDCircularGeometryXMLFile_h
#define __rtkThreeDCircularGeometryXMLFile_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "rtkXMLFile.h"
#include "rtkThreeDCircularGeometry.h"

namespace rtk
{

/** \class ThreeDCircularGeometryXMLFileReader
 * 
 * Reads an XML-format file containing geometry for reconstruction
 */
class ThreeDCircularGeometryXMLFileReader :
    public XMLFileReader< ThreeDCircularGeometry >
{
public:
  /** Standard typedefs */ 
  typedef ThreeDCircularGeometryXMLFileReader      Self;
  typedef XMLFileReader< ThreeDCircularGeometry >  Superclass;
  typedef itk::SmartPointer<Self>                  Pointer;

  /** Convenient typedefs */
  typedef ThreeDCircularGeometry                   GeometryType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ThreeDCircularGeometryXMLFileReader, XMLFileReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

protected:
  ThreeDCircularGeometryXMLFileReader():  m_Geometry(GeometryType::New()) {this->m_OutputObject = &(*m_Geometry);};
  virtual ~ThreeDCircularGeometryXMLFileReader() {};

  virtual void StartElement(const char * name);
  virtual void EndElement(const char *name);

private:
  ThreeDCircularGeometryXMLFileReader(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  GeometryType::Pointer m_Geometry;

  double m_RotationAngle, m_ProjectionOffsetX, m_ProjectionOffsetY;
};

/** \class ThreeDCircularGeometryXMLFileWriter
 * 
 * Writes an XML-format file containing geometry for reconstruction
 */
class ThreeDCircularGeometryXMLFileWriter :
    public XMLFileWriter< ThreeDCircularGeometry >
{
public:
  /** standard typedefs */
  typedef XMLFileWriter< ThreeDCircularGeometry >    Superclass;
  typedef ThreeDCircularGeometryXMLFileWriter        Self;
  typedef itk::SmartPointer<Self>                    Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ThreeDCircularGeometryXMLFileWriter, XMLFileWriter);

  /** Actually write out the file in question */
  virtual int WriteFile();

protected:
  ThreeDCircularGeometryXMLFileWriter() {};
  virtual ~ThreeDCircularGeometryXMLFileWriter() {};

private:
  ThreeDCircularGeometryXMLFileWriter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};
}

#endif
