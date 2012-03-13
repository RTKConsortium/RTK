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

  /** Latest version */
  static const unsigned int CurrentVersion = 1;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ThreeDCircularProjectionGeometryXMLFileReader, XMLFileReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Determine if a file can be read */
  int CanReadFile(const char* name);

protected:
  ThreeDCircularProjectionGeometryXMLFileReader();
  ~ThreeDCircularProjectionGeometryXMLFileReader() { };

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

  /** Projection parameters */
  double m_InPlaneAngle, m_OutOfPlaneAngle, m_GantryAngle;
  double m_SourceToIsocenterDistance, m_SourceOffsetX, m_SourceOffsetY;
  double m_SourceToDetectorDistance, m_ProjectionOffsetX, m_ProjectionOffsetY;

  /** Projection matrix */
  ThreeDCircularProjectionGeometry::MatrixType m_Matrix;

  /** File format version */
  unsigned int m_Version;
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
  
  /** If all values are equal in v, write first value (if not 0.) in
      output file with parameter value s and return true. Return false
      otherwise.
   */
  bool WriteGlobalParameter(std::ofstream &output, const std::string &indent,
                            const std::vector<double> &v, const std::string &s);

  /** Write projection specific parameter with name s. */
  void WriteLocalParameter(std::ofstream &output, const std::string &indent,
                           const double &v, const std::string &s);

private:
   //purposely not implemented
  ThreeDCircularProjectionGeometryXMLFileWriter(const Self&);
  void operator=(const Self&);

};
}

#endif
