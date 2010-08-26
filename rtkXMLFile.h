#ifndef __rtkXMLFile_h
#define __rtkXMLFile_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "itkXMLFile.h"

namespace rtk
{

/** \class XMLFileReader
 * 
 * Implements default common functions for itk::XMLFileReader
 */
template <class T>
class XMLFileReader :
    public itk::XMLReader< T >
{
public:
  /** Standard typedefs */ 
  typedef XMLFileReader                          Self;
  typedef itk::XMLReader< T >                    Superclass;
  typedef itk::SmartPointer<Self>                Pointer;

public:
  /** Determine if a file can be read */
  virtual int CanReadFile(const char* name);

  /** Callback function -- called from XML parser with start-of-element
   * information.
   */
  virtual void StartElement(const char * name,const char **atts);
  virtual void StartElement(const char * name) = 0;

protected:
  XMLFileReader(){};
  virtual ~XMLFileReader() {};

  virtual void CharacterDataHandler(const char *inData, int inLength);

  std::string                       m_CurCharacterData;

private:
  XMLFileReader(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

/** \class XMLFileWriter
 * 
 * Implements default common functions for itk::XMLFileWriter
 */
template <class T>
class XMLFileWriter :
    public itk::XMLWriterBase< T >
{
public:
  /** standard typedefs */
  typedef itk::XMLWriterBase< T >                    Superclass;
  typedef XMLFileWriter                              Self;
  typedef itk::SmartPointer<Self>                    Pointer;

  /** Test whether a file is writable. */
  virtual int CanWriteFile(const char* name);

protected:
  XMLFileWriter() {};
  virtual ~XMLFileWriter() {};

private:
  XMLFileWriter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};
}

#include "rtkXMLFile.txx"

#endif
