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
#include "srtkThreeDCircularProjectionGeometry.h"
#include "srtkTemplateFunctions.h"

#include "rtkThreeDCircularProjectionGeometry.h"

#include <memory>
#include "nsstd/type_traits.h"


namespace rtk
{
namespace simple
{
class PimpleThreeDCircularProjectionGeometry
{
public:
  typedef PimpleThreeDCircularProjectionGeometry  Self;
  typedef rtk::ThreeDCircularProjectionGeometry   ProjectionGeometryType;
  typedef ProjectionGeometryType::Pointer         ProjectionGeometryPointer;

  PimpleThreeDCircularProjectionGeometry( ProjectionGeometryType * p)
    {
    this->m_ProjectionGeometry = p;
    }

  PimpleThreeDCircularProjectionGeometry( )
    {
    this->m_ProjectionGeometry = ProjectionGeometryType::New();
    }

  PimpleThreeDCircularProjectionGeometry( Self &s )
    : m_ProjectionGeometry( s.m_ProjectionGeometry )
    {}

  PimpleThreeDCircularProjectionGeometry &operator=( const PimpleThreeDCircularProjectionGeometry &s )
    {
    m_ProjectionGeometry = s.m_ProjectionGeometry;
    }

  ProjectionGeometryType::Pointer GetProjectionGeometry( void ) { return this->m_ProjectionGeometry.GetPointer(); }
  ProjectionGeometryType::ConstPointer GetProjectionGeometry( void ) const { return this->m_ProjectionGeometry.GetPointer(); }

  PimpleThreeDCircularProjectionGeometry *ShallowCopy( void ) const
    {
    return new Self( this->m_ProjectionGeometry.GetPointer() );
    }

  PimpleThreeDCircularProjectionGeometry *DeepCopy( void ) const
    {
    PimpleThreeDCircularProjectionGeometry *copy( new Self( this->m_ProjectionGeometry->Clone() ) );
    return copy;
    }

  int GetReferenceCount( ) const
    {
    return this->m_ProjectionGeometry->GetReferenceCount();
    }

  std::string ToString( void ) const
    {
    std::ostringstream out;
    this->GetProjectionGeometry()->Print ( out );
    return out.str();
    }

  /** Add the projection */
  void AddProjection(float sid,float sdd,float angle,float isox=0.,float isoy=0., float oa=0., float ia=0., float sx=0., float sy=0.)
    {
    this->m_ProjectionGeometry->AddProjection(sid,sdd,angle,isox,isoy,oa,ia,sx,sy);
    }

private:

  ProjectionGeometryPointer m_ProjectionGeometry;
};

//
// class ThreeDCircularProjectionGeometry
//

ThreeDCircularProjectionGeometry::ThreeDCircularProjectionGeometry( )
  {
  m_PimpleThreeDCircularProjectionGeometry = new PimpleThreeDCircularProjectionGeometry();
  }

 ThreeDCircularProjectionGeometry::~ThreeDCircularProjectionGeometry()
  {
    delete m_PimpleThreeDCircularProjectionGeometry;
    this->m_PimpleThreeDCircularProjectionGeometry = NULL;
  }

ThreeDCircularProjectionGeometry::ThreeDCircularProjectionGeometry( const ThreeDCircularProjectionGeometry &txf )
    : m_PimpleThreeDCircularProjectionGeometry( NULL )
  {
   m_PimpleThreeDCircularProjectionGeometry= txf.m_PimpleThreeDCircularProjectionGeometry->ShallowCopy();
  }

ThreeDCircularProjectionGeometry& ThreeDCircularProjectionGeometry::operator=( const ThreeDCircularProjectionGeometry & txf )
  {
    // note: if txf and this are the same,the following statements
    // will be safe. It's also exception safe.
    std::auto_ptr<PimpleThreeDCircularProjectionGeometry> temp( txf.m_PimpleThreeDCircularProjectionGeometry->ShallowCopy() );
    delete this->m_PimpleThreeDCircularProjectionGeometry;
    this->m_PimpleThreeDCircularProjectionGeometry = temp.release();
    return *this;
  }

rtk::ProjectionGeometry<3>* ThreeDCircularProjectionGeometry::GetRTKBase ( void )
  {
    assert( m_PimpleThreeDCircularProjectionGeometry );
    return this->m_PimpleThreeDCircularProjectionGeometry->GetProjectionGeometry();
  }

const rtk::ProjectionGeometry<3>* ThreeDCircularProjectionGeometry::GetRTKBase ( void ) const
  {
    assert( m_PimpleThreeDCircularProjectionGeometry );
    return this->m_PimpleThreeDCircularProjectionGeometry->GetProjectionGeometry();
  }


void ThreeDCircularProjectionGeometry::AddProjection(float sid,float sdd,float angle,float isox,float isoy, float oa, float ia, float sx, float sy)
  {
  assert( m_PimpleThreeDCircularProjectionGeometry );
  this->m_PimpleThreeDCircularProjectionGeometry->AddProjection(sid,sdd,angle,isox,isoy,oa,ia,sx,sy);
  }
  
std::string ThreeDCircularProjectionGeometry::ToString( void ) const
  {
  assert( m_PimpleThreeDCircularProjectionGeometry );
  return this->m_PimpleThreeDCircularProjectionGeometry->ToString();
  }

}
}
