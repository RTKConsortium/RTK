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
#include "srtkThreeDimCircularProjectionGeometryXMLFileWriter.h"
#include <rtkThreeDCircularProjectionGeometryXMLFile.h>

namespace rtk {
namespace simple {

void WriteXML ( const ThreeDimCircularProjectionGeometry& geometry, const std::string &inFileName )
  {
  ThreeDimCircularProjectionGeometryXMLFileWriter writer;
  writer.Execute ( geometry, inFileName );
  }


ThreeDimCircularProjectionGeometryXMLFileWriter::ThreeDimCircularProjectionGeometryXMLFileWriter()
  {
  // This is probably not needed
  //this->m_MemberFactory.reset( new detail::MemberFunctionFactory<MemberFunctionType>( this ) );
  }

std::string ThreeDimCircularProjectionGeometryXMLFileWriter::ToString() const
  {
  std::ostringstream out;
  out << "rtk::simple::ThreeDimCircularProjectionGeometryXMLFileWriter";
  out << std::endl;

  out << "  FileName: \"";
  this->ToStringHelper(out, this->m_FileName);
  out << "\"" << std::endl;

  return out.str();
  }

ThreeDimCircularProjectionGeometryXMLFileWriter& ThreeDimCircularProjectionGeometryXMLFileWriter::SetFileName ( std::string fn )
  {
  this->m_FileName = fn;
  return *this;
  }

std::string ThreeDimCircularProjectionGeometryXMLFileWriter::GetFileName() const
  {
  return this->m_FileName;
  }

ThreeDimCircularProjectionGeometryXMLFileWriter& ThreeDimCircularProjectionGeometryXMLFileWriter
::Execute ( const ThreeDimCircularProjectionGeometry& geometry, const std::string &inFileName )
  {
  this->SetFileName( inFileName );
  return this->Execute( geometry );
  }

ThreeDimCircularProjectionGeometryXMLFileWriter& ThreeDimCircularProjectionGeometryXMLFileWriter
::Execute ( const ThreeDimCircularProjectionGeometry& geometry )
  {
  typedef rtk::ThreeDCircularProjectionGeometryXMLFileWriter Writer;
  Writer::Pointer writer = Writer::New();
  writer->SetFilename ( this->m_FileName.c_str() );

  writer->SetObject ( const_cast< ThreeDCircularProjectionGeometry * > (geometry.GetRTKBase()) );

  //this->PreUpdate( writer.GetPointer() );
  writer->WriteFile();

  return *this;
  }

} // end namespace simple
} // end namespace rtk
