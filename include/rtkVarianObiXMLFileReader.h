/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkVarianObiXMLFileReader_h
#define rtkVarianObiXMLFileReader_h

#ifdef _MSC_VER
#  pragma warning(disable : 4786)
#endif

#include <itkXMLFile.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>

#include "RTKExport.h"
#include "rtkMacro.h"

namespace rtk
{

/** \class VarianObiXMLFileReader
 *
 * Reads the XML-format file written by a Varian OBI
 * machine for every acquisition
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
class RTK_EXPORT VarianObiXMLFileReader : public itk::XMLReader<itk::MetaDataDictionary>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VarianObiXMLFileReader);

  /** Standard type alias */
  using Self = VarianObiXMLFileReader;
  using Superclass = itk::XMLReader<itk::MetaDataDictionary>;
  using Pointer = itk::SmartPointer<Self>;

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(VarianObiXMLFileReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Determine if a file can be read */
  int
  CanReadFile(const char * name) override;

protected:
  VarianObiXMLFileReader() { m_OutputObject = &m_Dictionary; };
  ~VarianObiXMLFileReader() override = default;

  void
  StartElement(const char * name, const char ** atts) override;

  void
  EndElement(const char * name) override;

  void
  CharacterDataHandler(const char * inData, int inLength) override;

private:
  itk::MetaDataDictionary m_Dictionary;
  std::string             m_CurCharacterData;
};

} // namespace rtk
#endif
