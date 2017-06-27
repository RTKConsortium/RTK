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
#pragma warning ( disable : 4786 )
#endif

#include "rtkWin32Header.h"
#include <itkXMLFile.h>
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

/** \class ElektaXVI5GeometryXMLFileReader
 *
 * Reads an XML-format file of XVI version = 5.0.2  (_Frame.xml in each projection directory). From XVI_v5 on, thre is no need of accessing .DBF files (FRAME.DBF / IMAGE.DBF).
 * This class is basically inspired by ThreeDCircularProjectionGeometryXMLFileReader.
 * Writer is not implemented.
 * SAD = 1000 mm, SID = 1536 mm are hard-coded since _Frame.xml doesn't include these values.
 * Regarding PanelOffset, XVI5 specifies position of the center (UCentre, VCentre) instead of offset.
 * Therefore, negation is required to get classical m_ProjectionOffsetX and m_ProjectionOffsetY values.
 *
 * \author Yang K Park (theday79@gmail.com)
 *
 * \ingroup IOFilters
 */
class RTK_EXPORT ElektaXVI5GeometryXMLFileReader :
  public itk::XMLReader< ThreeDCircularProjectionGeometry >
{
public:
  /** Standard typedefs */
  typedef ElektaXVI5GeometryXMLFileReader                    Self;
  typedef itk::XMLReader< ThreeDCircularProjectionGeometry > Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;

  /** Convenient typedefs */
  typedef ThreeDCircularProjectionGeometry GeometryType;
  typedef GeometryType::Pointer            GeometryPointer;

  /** Latest version */
  static const unsigned int CurrentVersion = 2;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ElektaXVI5GeometryXMLFileReader, itk::XMLFileReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Determine if a file can be read */
  int CanReadFile(const char* name) ITK_OVERRIDE;

  /** Get smart pointer to projection geometry. */
  itkGetMacro(Geometry, GeometryPointer);

protected:
  ElektaXVI5GeometryXMLFileReader();
  ~ElektaXVI5GeometryXMLFileReader() {}

  /** Callback function -- called from XML parser with start-of-element
   * information.
   */
  void StartElement(const char * name,const char **atts) ITK_OVERRIDE;

  void StartElement(const char * name);

  void EndElement(const char *name) ITK_OVERRIDE;

  void CharacterDataHandler(const char *inData, int inLength) ITK_OVERRIDE;

private:
   //purposely not implemented
  ElektaXVI5GeometryXMLFileReader(const Self&);
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

  /** Projection matrix */
  ThreeDCircularProjectionGeometry::MatrixType m_Matrix;

  /** File format version */
  unsigned int m_Version;
};
}

#endif
