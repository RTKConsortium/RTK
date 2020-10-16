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

#ifndef rtkOraXMLFileReader_h
#define rtkOraXMLFileReader_h

#include <itkXMLFile.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>

#include "rtkMacro.h"

namespace rtk
{

/** \class OraXMLFileReader
 *
 * Reads the XML-format file written by a medPhoton scanner
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
class OraXMLFileReader : public itk::XMLReader<itk::MetaDataDictionary>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(OraXMLFileReader);
#else
  ITK_DISALLOW_COPY_AND_MOVE(OraXMLFileReader);
#endif

  /** Standard type alias */
  using Self = OraXMLFileReader;
  using Superclass = itk::XMLReader<itk::MetaDataDictionary>;
  using Pointer = itk::SmartPointer<Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(OraXMLFileReader, itk::XMLReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Determine if a file can be read */
  int
  CanReadFile(const char * name) override;

protected:
  OraXMLFileReader();
  ~OraXMLFileReader() override = default;

  void
  StartElement(const char * name, const char ** atts) override;

  void
  EndElement(const char * name) override;

  void
  CharacterDataHandler(const char * inData, int inLength) override;

  void
  EncapsulatePoint(const char * metaName, const char * name);
  void
  EncapsulateMatrix3x3(const char * metaName, const char * name);
  void
  EncapsulateDouble(const char * metaName, const char * name);
  void
  EncapsulateString(const char * metaName, const char * name);

private:
  itk::MetaDataDictionary m_Dictionary;
  std::string             m_CurCharacterData;
};

} // namespace rtk
#endif
