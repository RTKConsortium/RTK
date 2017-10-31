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
#ifndef __srtkThreeDCircularProjectionGeometryXMLFileWriter_hxx
#define __srtkThreeDCircularProjectionGeometryXMLFileWriter_hxx

#include "srtkThreeDCircularProjectionGeometryXMLFileWriter.h"
#include <rtkThreeDCircularProjectionGeometryXMLFile.h>

namespace rtk {
namespace simple {

ThreeDCircularProjectionGeometryXMLFileWriter
::ThreeDCircularProjectionGeometryXMLFileWriter(void)
{
  this->m_Writer = rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
}

/** Set the filename to write */
void 
ThreeDCircularProjectionGeometryXMLFileWriter
::SetFilename(const std::string & _arg)
{
  this->m_Writer->SetFilename(_arg);
}

/** determine whether a file can be opened and read */
int
ThreeDCircularProjectionGeometryXMLFileWriter
::CanWriteFile(const char *name)
{
  return this->m_Writer->CanWriteFile(name);
}

void
ThreeDCircularProjectionGeometryXMLFileWriter
::Update()
{
  this->m_Writer->WriteFile();
}

void
ThreeDCircularProjectionGeometryXMLFileWriter
::SetInput(GeometryType* geometry)
{
  this->m_Writer->SetObject(geometry);
}

} // end namespace simple
} // end namespace rtk

#endif