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
#include "srtkThreeDimCircularProjectionGeometry.h"
#include "srtkTemplateFunctions.h"

#include "rtkThreeDCircularProjectionGeometry.h"

#include <memory>
#include "nsstd/type_traits.h"


namespace rtk
{
namespace simple
{

class PimpleThreeDimCircularProjectionGeometry
{
public:
  typedef PimpleThreeDimCircularProjectionGeometry  Self;
  typedef rtk::ThreeDCircularProjectionGeometry   ProjectionGeometryType;
  typedef ProjectionGeometryType::Pointer         ProjectionGeometryPointer;

  PimpleThreeDimCircularProjectionGeometry( ProjectionGeometryType * p)
    {
    this->m_ProjectionGeometry = p;
    }

  PimpleThreeDimCircularProjectionGeometry( )
    {
    this->m_ProjectionGeometry = ProjectionGeometryType::New();
    }

  PimpleThreeDimCircularProjectionGeometry( Self &s )
    : m_ProjectionGeometry( s.m_ProjectionGeometry )
    {}

  PimpleThreeDimCircularProjectionGeometry &operator=( const PimpleThreeDimCircularProjectionGeometry &s )
    {
    m_ProjectionGeometry = s.m_ProjectionGeometry;
    }

  ProjectionGeometryType::Pointer GetProjectionGeometry( void ) { return this->m_ProjectionGeometry.GetPointer(); }
  ProjectionGeometryType::ConstPointer GetProjectionGeometry( void ) const { return this->m_ProjectionGeometry.GetPointer(); }

  PimpleThreeDimCircularProjectionGeometry *ShallowCopy( void ) const
    {
      return new Self( this->m_ProjectionGeometry.GetPointer() );
    }

  PimpleThreeDimCircularProjectionGeometry *DeepCopy( void ) const
    {
    PimpleThreeDimCircularProjectionGeometry *copy( new Self( this->m_ProjectionGeometry->Clone() ) );
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
// class ThreeDimCircularProjectionGeometry
//

ThreeDimCircularProjectionGeometry::ThreeDimCircularProjectionGeometry( )
  {
  m_PimpleThreeDimCircularProjectionGeometry = new PimpleThreeDimCircularProjectionGeometry();
  }

 ThreeDimCircularProjectionGeometry::~ThreeDimCircularProjectionGeometry()
  {
    delete m_PimpleThreeDimCircularProjectionGeometry;
    this->m_PimpleThreeDimCircularProjectionGeometry = NULL;
  }

ThreeDimCircularProjectionGeometry::ThreeDimCircularProjectionGeometry( const ThreeDimCircularProjectionGeometry &txf )
    : m_PimpleThreeDimCircularProjectionGeometry( NULL )
  {
   m_PimpleThreeDimCircularProjectionGeometry= txf.m_PimpleThreeDimCircularProjectionGeometry->ShallowCopy();
  }

ThreeDimCircularProjectionGeometry& ThreeDimCircularProjectionGeometry::operator=( const ThreeDimCircularProjectionGeometry & txf )
  {
    // note: if txf and this are the same,the following statements
    // will be safe. It's also exception safe.
    std::auto_ptr<PimpleThreeDimCircularProjectionGeometry> temp( txf.m_PimpleThreeDimCircularProjectionGeometry->ShallowCopy() );
    delete this->m_PimpleThreeDimCircularProjectionGeometry;
    this->m_PimpleThreeDimCircularProjectionGeometry = temp.release();
    return *this;
  }

rtk::ThreeDCircularProjectionGeometry* ThreeDimCircularProjectionGeometry::GetRTKBase ( void )
  {
    assert( m_PimpleThreeDimCircularProjectionGeometry );
    return this->m_PimpleThreeDimCircularProjectionGeometry->GetProjectionGeometry();
  }

const rtk::ThreeDCircularProjectionGeometry* ThreeDimCircularProjectionGeometry::GetRTKBase ( void ) const
  {
    assert( m_PimpleThreeDimCircularProjectionGeometry );
    return this->m_PimpleThreeDimCircularProjectionGeometry->GetProjectionGeometry();
  }


void ThreeDimCircularProjectionGeometry::AddProjection(float sid,float sdd,float angle,float isox,float isoy, float oa, float ia, float sx, float sy)
  {
  assert( m_PimpleThreeDimCircularProjectionGeometry );
  this->m_PimpleThreeDimCircularProjectionGeometry->AddProjection(sid,sdd,angle,isox,isoy,oa,ia,sx,sy);
  }
  
std::string ThreeDimCircularProjectionGeometry::ToString( void ) const
  {
  assert( m_PimpleThreeDimCircularProjectionGeometry );
  return this->m_PimpleThreeDimCircularProjectionGeometry->ToString();
  }

}
}
