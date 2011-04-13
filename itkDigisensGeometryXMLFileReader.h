#ifndef __itkDigisensGeometryXMLFileReader_h
#define __itkDigisensGeometryXMLFileReader_h

#include <itkXMLFile.h>
#include <itkMetaDataDictionary.h>

namespace itk
{

/** \class itkDigisensGeometryXMLFileReader
 *
 * Reads the XML-format file written by Digisens geometric
 * calibration tool.
 */
class DigisensGeometryXMLFileReader : public XMLReader<MetaDataDictionary>
{
public:
  /** Standard typedefs */
  typedef DigisensGeometryXMLFileReader                      Self;
  typedef XMLReader<MetaDataDictionary>                      Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef enum {ROTATION,XRAY,CAMERA,RADIOS,GRID,PROCESSING} CurrentSectionType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(DigisensGeometryXMLFileReader, XMLReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Determine if a file can be read */
  int CanReadFile(const char* name);

protected:
  DigisensGeometryXMLFileReader(){
    m_OutputObject = &m_Dictionary; m_TreeLevel = 0;
  }
  virtual ~DigisensGeometryXMLFileReader() {
  }

  virtual void StartElement(const char * name,const char **atts);

  virtual void EndElement(const char *name);

  void CharacterDataHandler(const char *inData, int inLength);

private:
  DigisensGeometryXMLFileReader(const Self&); //purposely not implemented
  void operator=(const Self&);                //purposely not implemented

  MetaDataDictionary m_Dictionary;
  std::string        m_CurCharacterData;
  int                m_NumberOfFiles;
  CurrentSectionType m_CurrentSection;
  int                m_TreeLevel;
};

}
#endif
