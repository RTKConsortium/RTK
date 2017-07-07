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
#ifndef __srtkThreeDCircularProjectionGeometry_h
#define __srtkThreeDCircularProjectionGeometry_h

#include "srtkCommon.h"
#include "srtkExceptionObject.h"
#include <vector>

namespace rtk{
  template<unsigned int TDimension> class ProjectionGeometry;
}

namespace rtk
{
namespace simple
{

class PimpleThreeDCircularProjectionGeometry;

/** \class ThreeDCircularProjectionGeometry
 * \brief A simplified wrapper around RTK's ThreeDCircularProjectionGeometry.
 *
 */
class SRTKCommon_EXPORT ThreeDCircularProjectionGeometry
{
public:
  typedef ThreeDCircularProjectionGeometry Self;

  /** \brief 
   */
  ThreeDCircularProjectionGeometry( void );
  virtual ~ThreeDCircularProjectionGeometry( void );

  /** \brief Copy constructor and assignment operator
   * Performs a shallow copy of the internal RTK Object.
   * @{
   */
  ThreeDCircularProjectionGeometry &operator=( const ThreeDCircularProjectionGeometry & );
  ThreeDCircularProjectionGeometry( const ThreeDCircularProjectionGeometry & );
  /**@}*/


  /** Get access to internal RTK data object.
   * @{
   */
  rtk::ProjectionGeometry<3>* GetRTKBase( void );
  const rtk::ProjectionGeometry<3>* GetRTKBase( void ) const;
  /**@}*/

  const std::vector<double> &GetGantryAngles() const;
  const std::vector<double> &GetOutOfPlaneAngles() const;
  const std::vector<double> &GetInPlaneAngles() const;
  const std::vector<double> &GetSourceAngles() const;
  const std::vector<double>  GetTiltAngles() const;
  const std::vector<double> &GetSourceToIsocenterDistances() const;
  const std::vector<double> &GetSourceOffsetsX() const;
  const std::vector<double> &GetSourceOffsetsY() const;
  const std::vector<double> &GetSourceToDetectorDistances() const;
  const std::vector<double> &GetProjectionOffsetsX() const;
  const std::vector<double> &GetProjectionOffsetsY() const;
  const std::vector<double>  GetSourcePosition( const unsigned int i ) const;
  const std::vector<double>  GetRotationMatrix( const unsigned int i) const;
  const std::vector<double>  GetMatrix( const unsigned int i) const;
  const std::vector<double>  GetProjectionCoordinatesToFixedSystemMatrix( const unsigned int i) const;

  const double GetRadiusCylindricalDetector();
  void SetRadiusCylindricalDetector(const double radius);

  /** Add the projection with angles in degress
   * @{
   */
  void AddProjection(float sid, float sdd, float angle, float isox=0., float isoy=0., float oa=0., float ia=0., float sx=0., float sy=0.);
  /**@}*/

  /** Add the projection with angles in radians
   * @{
   */
  void AddProjectionInRadians(float sid, float sdd, float angle, float isox=0., float isoy=0., float oa=0., float ia=0., float sx=0., float sy=0.);
  /**@}*/

  /** Add the projection with projection matrix 
   * @{
   */
  void AddProjection(const std::vector<double> matrix);
  /**@}*/

  /** Add the projection with source and detector positions and orientations 
   * @{
   */
  void AddProjection(const std::vector<double> sourcePosition, const std::vector<double> detectorPosition, const std::vector<double> detectorRowVector, const std::vector<double> detectorColumnVector);
  /**@}*/

  /** Clear the geometry object
   * @{
   */
  void Clear();
  /**@}*/

  /** Return the current geometry as a string
   * @{
   */
  std::string ToString( void ) const;
  /**@}*/

protected:

private:

  // As is the architecture of all SimpleRTK pimples,
  // this pointer should never be null and should always point to a
  // valid object
  PimpleThreeDCircularProjectionGeometry *m_PimpleThreeDCircularProjectionGeometry;
};

}
}

#endif // __srtkThreeDCircularProjectionGeometry_h
