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

#ifndef rtkDigisensGeometryXMLFileReader_h
#define rtkDigisensGeometryXMLFileReader_h

#include "RTKExport.h"
#include <itkXMLFile.h>
#include <itkMetaDataDictionary.h>

#include "rtkMacro.h"

namespace rtk
{

/** \class DigisensGeometryXMLFileReader
 *
 * Reads the XML-format file written by Digisens geometric
 * calibration tool.
 *
 * \author Simon Rit
 *
 * \test rtkdigisenstest.cxx
 *
 * \ingroup RTK
 */
class RTK_EXPORT DigisensGeometryXMLFileReader : public itk::XMLReader<itk::MetaDataDictionary>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DigisensGeometryXMLFileReader);

  /** Standard type alias */
  using Self = DigisensGeometryXMLFileReader;
  using Superclass = itk::XMLReader<itk::MetaDataDictionary>;
  using Pointer = itk::SmartPointer<Self>;
  using CurrentSectionType = enum { NONE, ROTATION, XRAY, CAMERA, RADIOS, GRID, PROCESSING };

  /** Run-time type information (and related methods). */
#ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(DigisensGeometryXMLFileReader);
#else
  itkTypeMacro(DigisensGeometryXMLFileReader, itk::XMLReader);
#endif

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Determine if a file can be read */
  int
  CanReadFile(const char * name) override;

protected:
  DigisensGeometryXMLFileReader();
  ~DigisensGeometryXMLFileReader() override = default;

  void
  StartElement(const char * name, const char ** atts) override;

  void
  EndElement(const char * name) override;

  void
  CharacterDataHandler(const char * inData, int inLength) override;

private:
  itk::MetaDataDictionary m_Dictionary;
  std::string             m_CurCharacterData;
  int                     m_NumberOfFiles;
  CurrentSectionType      m_CurrentSection;
  int                     m_TreeLevel;
};

} // namespace rtk
#endif
