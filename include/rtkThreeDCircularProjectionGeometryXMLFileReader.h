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

#ifndef rtkThreeDCircularProjectionGeometryXMLFileReader_h
#define rtkThreeDCircularProjectionGeometryXMLFileReader_h

#ifdef _MSC_VER
#  pragma warning(disable : 4786)
#endif

#include <itkXMLFile.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "RTKExport.h"

namespace rtk
{

/** \class ThreeDCircularProjectionGeometryXMLFileReader
 *
 * Reads an XML-format file containing geometry for reconstruction
 *
 * \test rtkgeometryfiletest.cxx, rtkvariantest.cxx, rtkxradtest.cxx,
 * rtkdigisenstest.cxx, rtkelektatest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK IOFilters
 */
class RTK_EXPORT ThreeDCircularProjectionGeometryXMLFileReader : public itk::XMLReader<ThreeDCircularProjectionGeometry>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(ThreeDCircularProjectionGeometryXMLFileReader);
#else
  ITK_DISALLOW_COPY_AND_MOVE(ThreeDCircularProjectionGeometryXMLFileReader);
#endif

  /** Standard type alias */
  using Self = ThreeDCircularProjectionGeometryXMLFileReader;
  using Superclass = itk::XMLReader<ThreeDCircularProjectionGeometry>;
  using Pointer = itk::SmartPointer<Self>;

  /** Convenient type alias */
  using GeometryType = ThreeDCircularProjectionGeometry;
  using GeometryPointer = GeometryType::Pointer;

  /** Latest version */
  static const unsigned int CurrentVersion = 3;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ThreeDCircularProjectionGeometryXMLFileReader, itk::XMLFileReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Determine if a file can be read */
  int
  CanReadFile(const char * name) override;

  /** Get smart pointer to projection geometry. */
  itkGetModifiableObjectMacro(Geometry, GeometryType);

protected:
  ThreeDCircularProjectionGeometryXMLFileReader();
  ~ThreeDCircularProjectionGeometryXMLFileReader() override = default;

  /** Callback function -- called from XML parser with start-of-element
   * information.
   */
  void
  StartElement(const char * name, const char ** atts) override;

  void
  StartElement(const char * name);

  void
  EndElement(const char * name) override;

  void
  CharacterDataHandler(const char * inData, int inLength) override;

private:
  GeometryPointer m_Geometry{ GeometryType::New() };

  std::string m_CurCharacterData{ "" };

  /** Projection parameters */
  double m_InPlaneAngle{ 0. };
  double m_OutOfPlaneAngle{ 0. };
  double m_GantryAngle{ 0. };
  double m_SourceToIsocenterDistance{ 0. };
  double m_SourceOffsetX{ 0. };
  double m_SourceOffsetY{ 0. };
  double m_SourceToDetectorDistance{ 0. };
  double m_ProjectionOffsetX{ 0. };
  double m_ProjectionOffsetY{ 0. };
  double m_CollimationUInf{ std::numeric_limits<double>::max() };
  double m_CollimationUSup{ std::numeric_limits<double>::max() };
  double m_CollimationVInf{ std::numeric_limits<double>::max() };
  double m_CollimationVSup{ std::numeric_limits<double>::max() };

  /** Projection matrix */
  ThreeDCircularProjectionGeometry::MatrixType m_Matrix;

  /** File format version */
  unsigned int m_Version{ 0 };
};
} // namespace rtk
#endif
