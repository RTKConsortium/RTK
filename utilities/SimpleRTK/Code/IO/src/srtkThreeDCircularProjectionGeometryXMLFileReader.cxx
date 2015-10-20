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
#include "srtkThreeDCircularProjectionGeometryXMLFileReader.h"
#include <rtkThreeDCircularProjectionGeometryXMLFile.h>

namespace rtk {
namespace simple {

ThreeDCircularProjectionGeometry ReadXML ( const std::string &inFileName )
  {
  ThreeDCircularProjectionGeometryXMLFileReader reader;
  return reader.Execute ( inFileName );
  }


ThreeDCircularProjectionGeometryXMLFileReader::ThreeDCircularProjectionGeometryXMLFileReader()
  {
  // This is probably not needed
  //this->m_MemberFactory.reset( new detail::MemberFunctionFactory<MemberFunctionType>( this ) );
  }

std::string ThreeDCircularProjectionGeometryXMLFileReader::ToString() const
  {
  std::ostringstream out;
  out << "rtk::simple::ThreeDCircularProjectionGeometryXMLFileReader";
  out << std::endl;

  out << "  FileName: \"";
  this->ToStringHelper(out, this->m_FileName);
  out << "\"" << std::endl;

  return out.str();
  }

ThreeDCircularProjectionGeometryXMLFileReader& ThreeDCircularProjectionGeometryXMLFileReader::SetFileName ( std::string fn )
  {
  this->m_FileName = fn;
  return *this;
  }

std::string ThreeDCircularProjectionGeometryXMLFileReader::GetFileName() const
  {
  return this->m_FileName;
  }

ThreeDCircularProjectionGeometry ThreeDCircularProjectionGeometryXMLFileReader
::Execute ( const std::string &inFileName )
  {
  this->SetFileName( inFileName );
  return this->Execute();
  }

ThreeDCircularProjectionGeometry ThreeDCircularProjectionGeometryXMLFileReader
::Execute ()
  {
  typedef rtk::ThreeDCircularProjectionGeometryXMLFileReader Reader;
  Reader::Pointer reader = Reader::New();
  reader->SetFilename ( this->m_FileName.c_str() );
  reader->GenerateOutputInformation();
  m_Geometry.Clear();
  for(unsigned int i=0; i<reader->GetOutputObject()->GetGantryAngles().size(); i++)
    m_Geometry.AddProjectionInRadians(
          reader->GetOutputObject()->GetSourceToIsocenterDistances()[i],
          reader->GetOutputObject()->GetSourceToDetectorDistances()[i],
          reader->GetOutputObject()->GetGantryAngles()[i],
          reader->GetOutputObject()->GetProjectionOffsetsX()[i],
          reader->GetOutputObject()->GetProjectionOffsetsY()[i],
          reader->GetOutputObject()->GetOutOfPlaneAngles()[i],
          reader->GetOutputObject()->GetInPlaneAngles()[i],
          reader->GetOutputObject()->GetSourceOffsetsX()[i],
          reader->GetOutputObject()->GetSourceOffsetsY()[i]);
  return m_Geometry;
  }

} // end namespace simple
} // end namespace rtk
