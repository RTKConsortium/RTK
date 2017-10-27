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

#ifndef rtkProjectionGeometry_hxx
#define rtkProjectionGeometry_hxx

namespace rtk {

template< unsigned int TDimension >
void
ProjectionGeometry< TDimension >
::PrintSelf( std::ostream& os, itk::Indent indent ) const
{
  os << "List of projection matrices:" << std::endl;
  for(unsigned int i=0; i<m_Matrices.size(); i++)
    {
    os << indent << "Matrix #" << i << ": "
       << m_Matrices[i] << std::endl;
    }
}

template< unsigned int TDimension >
void
ProjectionGeometry< TDimension >
::Clear()
{
  m_Matrices.clear();
  this->Modified();
}

}

#endif // rtkProjectionGeometry_hxx
