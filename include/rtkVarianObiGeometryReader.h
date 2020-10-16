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

#ifndef rtkVarianObiGeometryReader_h
#define rtkVarianObiGeometryReader_h

#include <itkLightProcessObject.h>
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

/** \class VarianObiGeometryReader
 *
 * Creates a 3D circular geometry from Varian OBI data.
 *
 * \test rtkvariantest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK IOFilters
 */
class RTK_EXPORT VarianObiGeometryReader : public itk::LightProcessObject
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(VarianObiGeometryReader);
#else
  ITK_DISALLOW_COPY_AND_MOVE(VarianObiGeometryReader);
#endif

  /** Standard type alias */
  using Self = VarianObiGeometryReader;
  using Superclass = itk::LightProcessObject;
  using Pointer = itk::SmartPointer<Self>;

  /** Convenient type alias */
  using GeometryType = ThreeDCircularProjectionGeometry;
  using FileNamesContainer = std::vector<std::string>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(VarianObiGeometryReader, LightProcessObject);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Get the pointer to the generated geometry object. */
  itkGetMacro(Geometry, GeometryType::Pointer);

  /** Set the path to the Varian OBI xml file containing geometric information. */
  itkGetMacro(XMLFileName, std::string);
  itkSetMacro(XMLFileName, std::string);

  /** Set the vector of strings that contains the projection file names. Files
   * are processed in sequential order. */
  void
  SetProjectionsFileNames(const FileNamesContainer & name)
  {
    if (m_ProjectionsFileNames != name)
    {
      m_ProjectionsFileNames = name;
      this->Modified();
    }
  }
  const FileNamesContainer &
  GetProjectionsFileNames() const
  {
    return m_ProjectionsFileNames;
  }

protected:
  VarianObiGeometryReader();


private:
  void
  GenerateData() override;

  GeometryType::Pointer m_Geometry;
  std::string           m_XMLFileName;
  FileNamesContainer    m_ProjectionsFileNames;
};

} // namespace rtk
#endif
