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
#include "srtkThreeDCircularProjectionGeometryXMLFileWriter.h"
#include <rtkThreeDCircularProjectionGeometryXMLFile.h>

namespace rtk {
namespace simple {

void WriteXML ( const ThreeDCircularProjectionGeometry& geometry, const std::string &inFileName )
  {
  ThreeDCircularProjectionGeometryXMLFileWriter writer;
  writer.Execute ( geometry, inFileName );
  }


ThreeDCircularProjectionGeometryXMLFileWriter::ThreeDCircularProjectionGeometryXMLFileWriter()
  {
  // This is probably not needed
  //this->m_MemberFactory.reset( new detail::MemberFunctionFactory<MemberFunctionType>( this ) );
  }

std::string ThreeDCircularProjectionGeometryXMLFileWriter::ToString() const
  {
  std::ostringstream out;
  out << "rtk::simple::ThreeDCircularProjectionGeometryXMLFileWriter";
  out << std::endl;

  out << "  FileName: \"";
  this->ToStringHelper(out, this->m_FileName);
  out << "\"" << std::endl;

  return out.str();
  }

ThreeDCircularProjectionGeometryXMLFileWriter& ThreeDCircularProjectionGeometryXMLFileWriter::SetFileName ( std::string fn )
  {
  this->m_FileName = fn;
  return *this;
  }

std::string ThreeDCircularProjectionGeometryXMLFileWriter::GetFileName() const
  {
  return this->m_FileName;
  }

ThreeDCircularProjectionGeometryXMLFileWriter& ThreeDCircularProjectionGeometryXMLFileWriter
::Execute ( const ThreeDCircularProjectionGeometry& geometry, const std::string &inFileName )
  {
  this->SetFileName( inFileName );
  return this->Execute( geometry );
  }

ThreeDCircularProjectionGeometryXMLFileWriter& ThreeDCircularProjectionGeometryXMLFileWriter
::Execute ( const ThreeDCircularProjectionGeometry& geometry )
  {
  typedef rtk::ThreeDCircularProjectionGeometryXMLFileWriter Writer;
  Writer::Pointer writer = Writer::New();
  writer->SetFilename ( this->m_FileName.c_str() );

  writer->SetObject ( const_cast<rtk::ThreeDCircularProjectionGeometry *>(dynamic_cast<const rtk::ThreeDCircularProjectionGeometry * > (geometry.GetRTKBase())) );
  writer->WriteFile();

  return *this;
  }

} // end namespace simple
} // end namespace rtk
