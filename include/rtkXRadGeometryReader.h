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

#ifndef rtkXRadGeometryReader_h
#define rtkXRadGeometryReader_h

#include <itkLightProcessObject.h>
#include "rtkReg23ProjectionGeometry.h"
#include "RTKExport.h"

namespace rtk
{

/** \class XRadGeometryReader
 *
 * Creates a 3D circular geometry from an exported XRad sinogram.
 *
 * \test rtkxradtest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK IOFilters
 */
class RTK_EXPORT XRadGeometryReader : public itk::LightProcessObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(XRadGeometryReader);

  /** Standard type alias */
  using Self = XRadGeometryReader;
  using Superclass = itk::LightProcessObject;
  using Pointer = itk::SmartPointer<Self>;

  /** Convenient type alias */
  using GeometryType = Reg23ProjectionGeometry;

  /** Run-time type information (and related methods). */
  itkTypeMacro(XRadGeometryReader, LightProcessObject);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Get the pointer to the generated geometry object. */
  itkGetMacro(Geometry, GeometryType::Pointer);

  /** Set the path to the header file containing the sinogram information */
  itkGetMacro(ImageFileName, std::string);
  itkSetMacro(ImageFileName, std::string);

protected:
  XRadGeometryReader();

private:
  void
  GenerateData() override;

  GeometryType::Pointer m_Geometry;
  std::string           m_ImageFileName;
};

} // namespace rtk
#endif
