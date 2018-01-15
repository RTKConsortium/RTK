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
#include "rtkWin32Header.h"
#include <gdcmDataSet.h>

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
 * \ingroup IOFilters
 */
class RTK_EXPORT BioscanGeometryReader:
    public itk::LightProcessObject
{
public:
  /** Standard typedefs */
  typedef BioscanGeometryReader   Self;
  typedef itk::LightProcessObject Superclass;
  typedef itk::SmartPointer<Self> Pointer;

  /** Convenient typedefs */
  typedef ThreeDCircularProjectionGeometry GeometryType;
  typedef GeometryType::Pointer            GeometryPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(BioscanGeometryReader, itk::LightProcessObject);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Get the pointer to the generated geometry object. */
  itkGetObjectMacro(Geometry, GeometryType);

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

protected:
  std::vector<float> GetVectorTagValue(const gdcm::DataSet & ds, uint16_t group, uint16_t element) const;
  std::string GetStringTagValue(const gdcm::DataSet & ds, uint16_t group, uint16_t element) const;
  double GetFloatTagValue(const gdcm::DataSet & ds, uint16_t group, uint16_t element) const;

  BioscanGeometryReader(): m_Geometry(ITK_NULLPTR) {};
  ~BioscanGeometryReader() {}

private:
  //purposely not implemented
  BioscanGeometryReader(const Self&);
  void operator=(const Self&);

  void GenerateData() ITK_OVERRIDE;

  GeometryPointer    m_Geometry;
  FileNamesContainer m_ProjectionsFileNames;
};

}

#endif // rtkBioscanGeometryReader_h
