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
#pragma warning ( disable : 4786 )
#endif

#include "rtkWin32Header.h"
#include <itkXMLFile.h>
#include "rtkThreeDCircularProjectionGeometry.h"

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
class RTK_EXPORT ThreeDCircularProjectionGeometryXMLFileReader :
  public itk::XMLReader< ThreeDCircularProjectionGeometry >
{
public:
  /** Standard typedefs */
  typedef ThreeDCircularProjectionGeometryXMLFileReader      Self;
  typedef itk::XMLReader< ThreeDCircularProjectionGeometry > Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;

  /** Convenient typedefs */
  typedef ThreeDCircularProjectionGeometry GeometryType;
  typedef GeometryType::Pointer            GeometryPointer;

  /** Latest version */
  static const unsigned int CurrentVersion = 3;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ThreeDCircularProjectionGeometryXMLFileReader, itk::XMLFileReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Determine if a file can be read */
  int CanReadFile(const char* name) ITK_OVERRIDE;

  /** Get smart pointer to projection geometry. */
  itkGetModifiableObjectMacro(Geometry, GeometryType);

protected:
  ThreeDCircularProjectionGeometryXMLFileReader();
  virtual ~ThreeDCircularProjectionGeometryXMLFileReader() ITK_OVERRIDE {}

  /** Callback function -- called from XML parser with start-of-element
   * information.
   */
  void StartElement(const char * name,const char **atts) ITK_OVERRIDE;

  void StartElement(const char * name);

  void EndElement(const char *name) ITK_OVERRIDE;

  void CharacterDataHandler(const char *inData, int inLength) ITK_OVERRIDE;

private:
   //purposely not implemented
  ThreeDCircularProjectionGeometryXMLFileReader(const Self&);
  void operator=(const Self&);

  GeometryPointer m_Geometry;

  std::string m_CurCharacterData;

  /** Projection parameters */
  double m_InPlaneAngle;
  double m_OutOfPlaneAngle;
  double m_GantryAngle;
  double m_SourceToIsocenterDistance;
  double m_SourceOffsetX;
  double m_SourceOffsetY;
  double m_SourceToDetectorDistance;
  double m_ProjectionOffsetX;
  double m_ProjectionOffsetY;
  double m_CollimationUInf;
  double m_CollimationUSup;
  double m_CollimationVInf;
  double m_CollimationVSup;

  /** Projection matrix */
  ThreeDCircularProjectionGeometry::MatrixType m_Matrix;

  /** File format version */
  unsigned int m_Version;
};
}
#endif
