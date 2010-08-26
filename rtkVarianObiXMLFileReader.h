#ifndef __rtkVarianObiXMLFileReader_h
#define __rtkVarianObiXMLFileReader_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "rtkXMLFile.h"
#include <itkMetaDataDictionary.h>

#include <map>

namespace rtk
{

/** \class rtkVarianObiXMLFileReader
 * 
 * Reads the XML-format file written by a Varian OBI
 * machine for every acquisition
 */
class VarianObiXMLFileReader :
    public XMLFileReader<itk::MetaDataDictionary>
{
public:
  /** Standard typedefs */
  typedef VarianObiXMLFileReader                  Self;
  typedef XMLFileReader<itk::MetaDataDictionary> Superclass;
  typedef itk::SmartPointer<Self>                 Pointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(VarianObiXMLFileReader, XMLFileReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

protected:
  VarianObiXMLFileReader(){m_OutputObject = &m_Dictionary;};
  virtual ~VarianObiXMLFileReader() {};

  virtual void StartElement(const char * name);
  virtual void EndElement(const char *name);

private:
  VarianObiXMLFileReader(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

  itk::MetaDataDictionary           m_Dictionary;
};

}
#endif
