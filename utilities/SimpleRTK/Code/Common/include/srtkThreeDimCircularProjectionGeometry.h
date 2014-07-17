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
#ifndef __srtkThreeDimCircularProjectionGeometry_h
#define __srtkThreeDimCircularProjectionGeometry_h

#include "srtkCommon.h"
#include "srtkExceptionObject.h"
#include <vector>

namespace rtk{
  class ThreeDCircularProjectionGeometry;
}

namespace rtk
{
namespace simple
{

class PimpleThreeDimCircularProjectionGeometry;

/** \class ThreeDimCircularProjectionGeometry
 * \brief A simplified wrapper around a variety of RTK ThreeDimCircularProjectionGeometry.
 *
 */
class SRTKCommon_EXPORT ThreeDimCircularProjectionGeometry
{
public:
  typedef ThreeDimCircularProjectionGeometry Self;

  /** \brief By default a 3-d identity transform is constructed
   */
  ThreeDimCircularProjectionGeometry( void );
  virtual ~ThreeDimCircularProjectionGeometry( void );

  /** \brief Copy constructor and assignment operator
   *
   * Performs a shallow copy of the internal ITK transform. A deep
   * copy will be done if the transform in modified.
   * @{
   */
  ThreeDimCircularProjectionGeometry &operator=( const ThreeDimCircularProjectionGeometry & );
  ThreeDimCircularProjectionGeometry( const ThreeDimCircularProjectionGeometry & );
  /**@}*/


  /** Get access to internal ITK data object.
   *
   * The return value should imediately be assigned to as
   * itk::SmartPointer.
   *
   * In many cases the value may need to be dynamically casted to
   * the the actual transform type.
   *
   * @{
   */
  rtk::ThreeDCircularProjectionGeometry* GetRTKBase( void );
  const rtk::ThreeDCircularProjectionGeometry* GetRTKBase( void ) const;
  /**@}*/

  // todo get transform type

  /** Add the projection
   * @{
   */
  void AddProjection(float sid,float sdd,float angle,float isox,float isoy);
  /**@}*/


  // todo set identity
  std::string ToString( void ) const;


protected:

private:

  // As is the architecture of all SimpleRTK pimples,
  // this pointer should never be null and should always point to a
  // valid object
  PimpleThreeDimCircularProjectionGeometry *m_PimpleThreeDimCircularProjectionGeometry;
};

}
}

#endif // __srtkThreeDimCircularProjectionGeometry_h
