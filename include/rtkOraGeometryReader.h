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

#ifndef rtkOraGeometryReader_h
#define rtkOraGeometryReader_h

#include <itkLightProcessObject.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkWin32Header.h"

namespace rtk
{

/** \class OraGeometryReader
 *
 * Creates a 3D circular geometry from an ora (medPhoton) dataset.
 *
 * \test rtkoratest
 *
 * \author Simon Rit
 *
 * \ingroup IOFilters
 */
class RTK_EXPORT OraGeometryReader : public itk::LightProcessObject
{
public:
  /** Standard typedefs */
  typedef OraGeometryReader       Self;
  typedef itk::LightProcessObject Superclass;
  typedef itk::SmartPointer<Self> Pointer;

  /** Convenient typedefs */
  typedef ThreeDCircularProjectionGeometry GeometryType;
  typedef GeometryType::PointType          PointType;
  typedef GeometryType::Matrix3x3Type      Matrix3x3Type;
  typedef GeometryType::VectorType         VectorType;
  typedef itk::Vector<double, 4>           MarginVectorType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(OraGeometryReader, itk::LightProcessObject);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Get the pointer to the generated geometry object. */
  itkGetMacro(Geometry, GeometryType::Pointer);

  /** Some convenient typedefs. */
  typedef std::vector<std::string>            FileNamesContainer;

  /** Set the vector of strings that contains the projection file names. Files
   * are processed in sequential order. */
  void SetProjectionsFileNames (const FileNamesContainer &name)
    {
    if ( m_ProjectionsFileNames != name)
      {
      m_ProjectionsFileNames = name;
      this->Modified();
      }
    }
  const FileNamesContainer & GetProjectionsFileNames() const
    {
    return m_ProjectionsFileNames;
    }

  /** Collimation margin: adds a small margin to the collimation edge to remove
   * collimator shadow. A positive value means less collimation. Default is 0.
   * The order is uinf, usup, vinf, vsup.
   * */
  itkGetMacro(CollimationMargin, MarginVectorType)
  itkSetMacro(CollimationMargin, MarginVectorType)


protected:
  OraGeometryReader(): m_Geometry(ITK_NULLPTR), m_CollimationMargin(0.) {};

  ~OraGeometryReader() {}

private:
  //purposely not implemented
  OraGeometryReader(const Self&);
  void operator=(const Self&);

  void GenerateData() ITK_OVERRIDE;

  GeometryType::Pointer m_Geometry;
  FileNamesContainer    m_ProjectionsFileNames;
  MarginVectorType      m_CollimationMargin;
};

}

#endif // rtkOraGeometryReader_h
