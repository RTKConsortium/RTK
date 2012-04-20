#ifndef __rtkVarianObiXMLFileReader_h
#define __rtkVarianObiXMLFileReader_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include <itkXMLFile.h>
#include <itkMetaDataDictionary.h>

#include <map>

namespace rtk
{

/** \class VarianObiXMLFileReader
 *
 * Reads the XML-format file written by a Varian OBI
 * machine for every acquisition
 */
class VarianObiXMLFileReader : public itk::XMLReader<itk::MetaDataDictionary>
{
public:
  /** Standard typedefs */
  typedef VarianObiXMLFileReader                  Self;
  typedef itk::XMLReader<itk::MetaDataDictionary> Superclass;
  typedef itk::SmartPointer<Self>                 Pointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(VarianObiXMLFileReader, itk::XMLReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Determine if a file can be read */
  int CanReadFile(const char* name);

protected:
  VarianObiXMLFileReader(){m_OutputObject = &m_Dictionary;};
  virtual ~VarianObiXMLFileReader() {};

  virtual void StartElement(const char * name,const char **atts);

  virtual void EndElement(const char *name);

  void CharacterDataHandler(const char *inData, int inLength);

private:
  VarianObiXMLFileReader(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

  itk::MetaDataDictionary m_Dictionary;
  std::string             m_CurCharacterData;
};

}
#endif
