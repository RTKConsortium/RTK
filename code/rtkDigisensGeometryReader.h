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

#ifndef rtkDigisensGeometryReader_h
#define rtkDigisensGeometryReader_h

#include <itkLightProcessObject.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkWin32Header.h"

namespace rtk
{

/** \class DigisensGeometryReader
 *
 * Creates a 3D circular geometry from an xml file created by the calibration
 * software developed by the Digisens company.
 *
 * \test rtkdigisenstest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup IOFilters
 */
class RTK_EXPORT DigisensGeometryReader :
  public itk::LightProcessObject
{
public:
  /** Standard typedefs */
  typedef DigisensGeometryReader  Self;
  typedef itk::LightProcessObject Superclass;
  typedef itk::SmartPointer<Self> Pointer;

  /** Convenient typedefs */
  typedef ThreeDCircularProjectionGeometry GeometryType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(DigisensGeometryReader, LightProcessObject);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Get the pointer to the generated geometry object. */
  itkGetMacro(Geometry, GeometryType::Pointer);

  /** Set the path to the file XML calibration file */
  itkGetMacro(XMLFileName, std::string);
  itkSetMacro(XMLFileName, std::string);

protected:
  DigisensGeometryReader();


private:
  //purposely not implemented
  DigisensGeometryReader(const Self&);
  void operator=(const Self&);

  void GenerateData() ITK_OVERRIDE;

  GeometryType::Pointer m_Geometry;
  std::string           m_XMLFileName;
};

}
#endif
