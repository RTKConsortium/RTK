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

#ifndef rtkBioscanGeometryReader_h
#define rtkBioscanGeometryReader_h

#include <itkLightProcessObject.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "RTKExport.h"

// Trick KWStyle with a first declaration of namespace RTK. Style test would
// not pass otherwise on the gdcm namespace.
namespace rtk
{}

// Forward declare class Dataset. This is done to avoid define conflicts in
// GDCM with lp_solve
namespace gdcm
{
class DataSet;
}

namespace rtk
{

/** \class BioscanGeometryReader
 *
 * Creates a 3D circular geometry from a Bioscan NanoSPECT/CT.
 *
 * \test rtkbioscantest
 *
 * \author Simon Rit
 *
 * \ingroup RTK IOFilters
 */

class RTK_EXPORT BioscanGeometryReader : public itk::LightProcessObject
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(BioscanGeometryReader);
#else
  ITK_DISALLOW_COPY_AND_MOVE(BioscanGeometryReader);
#endif

  /** Standard type alias */
  using Self = BioscanGeometryReader;
  using Superclass = itk::LightProcessObject;
  using Pointer = itk::SmartPointer<Self>;

  /** Convenient type alias */
  using GeometryType = ThreeDCircularProjectionGeometry;
  using GeometryPointer = GeometryType::Pointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(BioscanGeometryReader, itk::LightProcessObject);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Get the pointer to the generated geometry object. */
  itkGetModifiableObjectMacro(Geometry, GeometryType);

  /** Some convenient type alias. */
  using FileNamesContainer = std::vector<std::string>;

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
  std::vector<float>
  GetVectorTagValue(const gdcm::DataSet & ds, itk::uint16_t group, itk::uint16_t element) const;
  std::string
  GetStringTagValue(const gdcm::DataSet & ds, itk::uint16_t group, itk::uint16_t element) const;
  double
  GetFloatTagValue(const gdcm::DataSet & ds, itk::uint16_t group, itk::uint16_t element) const;

  BioscanGeometryReader()
    : m_Geometry(nullptr){};
  ~BioscanGeometryReader() override = default;

private:
  void
  GenerateData() override;

  GeometryPointer    m_Geometry;
  FileNamesContainer m_ProjectionsFileNames;
};

} // namespace rtk

#endif // rtkBioscanGeometryReader_h
