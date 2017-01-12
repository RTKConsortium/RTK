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

#ifndef rtkOraGeometryReader_hxx
#define rtkOraGeometryReader_hxx

#include "rtkMacro.h"
#include "rtkOraGeometryReader.h"

#include <itkDOMNodeXMLReader.h>

namespace rtk
{

OraGeometryReader::PointType
OraGeometryReader::ReadPointFromString(std::string s)
{
  GeometryType::PointType p;
  std::istringstream iss(s);
  for(int i=0; i<3; i++)
    {
    iss >> p[i];
    iss.ignore(1);
    }
  return p;
}

OraGeometryReader::Matrix3x3Type
OraGeometryReader::ReadMatrix3x3FromString(std::string s)
{
  Matrix3x3Type m;
  std::istringstream iss(s);
  for(int i=0; i<3; i++)
    {
    for(int j=0; j<3; j++)
      {
      iss >> m[i][j];
      iss.ignore(1);
      }
    }
  return m;
}

void OraGeometryReader::GenerateData()
{
  m_Geometry = GeometryType::New();

  for(unsigned int noProj=0; noProj < m_ProjectionsFileNames.size(); noProj++)
    {
    itk::DOMNodeXMLReader::Pointer parser = itk::DOMNodeXMLReader::New();
    parser->SetFileName(m_ProjectionsFileNames[noProj]);
    parser->Update();

    // Find volume_meta_info section
    const itk::DOMNode *vmi = parser->GetOutput()->GetChild("volume_meta_info");
    if(vmi == ITK_NULLPTR)
      vmi = parser->GetOutput()->GetChild("VOLUME_META_INFO");
    if(vmi == ITK_NULLPTR)
      itkExceptionMacro(<< "No volume_meta_info or VOLUME_META_INFO in " << m_ProjectionsFileNames[noProj]);

    // Find basic section in volume_meta_info section
    const itk::DOMNode *basic = vmi->GetChild("basic");
    if(basic == ITK_NULLPTR)
      basic = vmi->GetChild("BASIC");
    if(basic == ITK_NULLPTR)
      itkExceptionMacro(<< "No basic or BASIC in " << m_ProjectionsFileNames[noProj]);

    // Find source position in basic
    const itk::DOMNode *spx = basic->GetChild("sourceposition");
    if(spx == ITK_NULLPTR)
      spx = basic->GetChild("SourcePosition");
    if(spx == ITK_NULLPTR)
      itkExceptionMacro(<< "No sourceposition or SourcePosition in " << m_ProjectionsFileNames[noProj]);
    PointType sp = ReadPointFromString(spx->GetTextChild()->GetText());

    // Find detector position in basic
    const itk::DOMNode *dpx = basic->GetChild("origin");
    if(dpx == ITK_NULLPTR)
      dpx = basic->GetChild("Origin");
    if(dpx == ITK_NULLPTR)
      itkExceptionMacro(<< "No origin or Origin in " << m_ProjectionsFileNames[noProj]);
    PointType dp = ReadPointFromString(dpx->GetTextChild()->GetText());

    // Find detector direction in basic
    const itk::DOMNode *matx = basic->GetChild("direction");
    if(matx == ITK_NULLPTR)
      matx = basic->GetChild("Direction");
    if(matx == ITK_NULLPTR)
      itkExceptionMacro(<< "No direction or Direction in " << m_ProjectionsFileNames[noProj]);
    Matrix3x3Type mat = ReadMatrix3x3FromString(matx->GetTextChild()->GetText());

    // Got it, add to geometry
    m_Geometry->AddProjection(sp,
                              dp,
                              VectorType( &(mat[0][0]) ),
                              VectorType( &(mat[1][0]) ) );
    }
}
} //namespace rtk
#endif
