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
#ifndef __srtkThreeDCircularProjectionGeometryXMLFileReader_hxx
#define __srtkThreeDCircularProjectionGeometryXMLFileReader_hxx

#include "srtkThreeDCircularProjectionGeometryXMLFileReader.h"
#include <rtkThreeDCircularProjectionGeometryXMLFile.h>

namespace rtk {
namespace simple {

ThreeDCircularProjectionGeometryXMLFileReader
::ThreeDCircularProjectionGeometryXMLFileReader(void)
{
  this->m_Reader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
}

/** Set the filename to write */
void 
ThreeDCircularProjectionGeometryXMLFileReader
::SetFilename(const std::string & _arg)
{
  this->m_Reader->SetFilename(_arg);
}
/** Get the filename to write */
const char*
ThreeDCircularProjectionGeometryXMLFileReader
::GetFilename() const
{
  return this->m_Reader->GetFilename();
}

/** determine whether a file can be opened and read */
int
ThreeDCircularProjectionGeometryXMLFileReader
::CanReadFile(const char *name)
{
  return this->m_Reader->CanReadFile(name);
}

void
ThreeDCircularProjectionGeometryXMLFileReader
::Update()
{
  this->m_Reader->GenerateOutputInformation();
}

ThreeDCircularProjectionGeometryXMLFileReader::GeometryPointer
ThreeDCircularProjectionGeometryXMLFileReader
::GetOutput()
{
  GeometryPointer geometry = GeometryType::New();
  for (unsigned int i = 0; i < this->m_Reader->GetOutputObject()->GetGantryAngles().size(); i++)
  {
    geometry->AddProjectionInRadians(
      this->m_Reader->GetOutputObject()->GetSourceToIsocenterDistances()[i],
      this->m_Reader->GetOutputObject()->GetSourceToDetectorDistances()[i],
      this->m_Reader->GetOutputObject()->GetGantryAngles()[i],
      this->m_Reader->GetOutputObject()->GetProjectionOffsetsX()[i],
      this->m_Reader->GetOutputObject()->GetProjectionOffsetsY()[i],
      this->m_Reader->GetOutputObject()->GetOutOfPlaneAngles()[i],
      this->m_Reader->GetOutputObject()->GetInPlaneAngles()[i],
      this->m_Reader->GetOutputObject()->GetSourceOffsetsX()[i],
      this->m_Reader->GetOutputObject()->GetSourceOffsetsY()[i]);
  }
  return geometry;
}

} // end namespace simple
} // end namespace rtk

#endif