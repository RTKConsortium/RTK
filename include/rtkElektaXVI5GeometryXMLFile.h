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

#ifndef rtkElektaXVI5GeometryXMLFile_h
#define rtkElektaXVI5GeometryXMLFile_h

#ifdef _MSC_VER
#  pragma warning(disable : 4786)
#endif

#include "RTKExport.h"
#include <itkXMLFile.h>
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

/** \class ElektaXVI5GeometryXMLFileReader
 *
 * Reads an XML-format file of XVI version = 5.0.2  (_Frame.xml in each projection directory). From XVI_v5 on, thre is
 * no need of accessing .DBF files (FRAME.DBF / IMAGE.DBF). This class is basically inspired by
 * ThreeDCircularProjectionGeometryXMLFileReader. Writer is not implemented. SAD = 1000 mm, SID = 1536 mm are hard-coded
 * since _Frame.xml doesn't include these values. Regarding PanelOffset, XVI5 specifies position of the center (UCentre,
 * VCentre) instead of offset. Therefore, negation is required to get classical m_ProjectionOffsetX and
 * m_ProjectionOffsetY values.
 *
 * \author Yang K Park (theday79@gmail.com)
 *
 * \ingroup RTK IOFilters
 */
class RTK_EXPORT ElektaXVI5GeometryXMLFileReader : public itk::XMLReader<ThreeDCircularProjectionGeometry>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(ElektaXVI5GeometryXMLFileReader);
#else
  ITK_DISALLOW_COPY_AND_MOVE(ElektaXVI5GeometryXMLFileReader);
#endif

  /** Standard type alias */
  using Self = ElektaXVI5GeometryXMLFileReader;
  using Superclass = itk::XMLReader<ThreeDCircularProjectionGeometry>;
  using Pointer = itk::SmartPointer<Self>;

  /** Convenient type alias */
  using GeometryType = ThreeDCircularProjectionGeometry;
  using GeometryPointer = GeometryType::Pointer;

  /** Latest version */
  static const unsigned int CurrentVersion = 2;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ElektaXVI5GeometryXMLFileReader, itk::XMLFileReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Determine if a file can be read */
  int
  CanReadFile(const char * name) override;

  /** Get smart pointer to projection geometry. */
  itkGetMacro(Geometry, GeometryPointer);

protected:
  ElektaXVI5GeometryXMLFileReader();
  ~ElektaXVI5GeometryXMLFileReader() override = default;

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

  std::string m_CurCharacterData;

  /** Projection parameters */
  double m_InPlaneAngle{ 0. };
  double m_OutOfPlaneAngle{ 0. };
  double m_GantryAngle{ 0. };
  double m_SourceToIsocenterDistance{ 1000. };
  double m_SourceOffsetX{ 0. };
  double m_SourceOffsetY{ 0. };
  double m_SourceToDetectorDistance{ 1536. };
  double m_ProjectionOffsetX{ 0. };
  double m_ProjectionOffsetY{ 0. };

  /** Projection matrix */
  ThreeDCircularProjectionGeometry::MatrixType m_Matrix;
};
} // namespace rtk

#endif
