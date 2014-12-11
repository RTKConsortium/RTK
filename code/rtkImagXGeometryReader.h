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

#ifndef __rtkImagXGeometryReader_h
#define __rtkImagXGeometryReader_h

#include <itkLightProcessObject.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkWin32Header.h"

namespace rtk
{

/** \class ImagXGeometryReader
 *
 * Creates a 3D circular geometry from the IBA data set.
 *
 * \test rtkimagxtest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup IOFilters
 */
template< typename TInputImage >
class RTK_EXPORT ImagXGeometryReader : public itk::LightProcessObject
{
public:
  /** Standard typedefs */
  typedef ImagXGeometryReader         Self;
  typedef itk::LightProcessObject     Superclass;
  typedef itk::SmartPointer<Self>     Pointer;

  /** Convenient typedefs */
  typedef ThreeDCircularProjectionGeometry GeometryType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImagXGeometryReader, itk::LightProcessObject);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Get the pointer to the generated geometry object. */
  itkGetMacro(Geometry, GeometryType::Pointer);

  /** Some convenient typedefs. */
  typedef TInputImage                         InputImageType;
  typedef typename InputImageType::Pointer    InputImagePointer;
  typedef typename InputImageType::RegionType InputImageRegionType;
  typedef typename InputImageType::PixelType  InputImagePixelType;
  typedef std::vector<std::string>            FileNamesContainer;

  /** Set the iMagX calibration xml file*/
  itkGetMacro(CalibrationXMLFileName, std::string);
  itkSetMacro(CalibrationXMLFileName, std::string);

  /** Set the iMagX room setup xml file*/
  itkGetMacro(RoomXMLFileName, std::string);
  itkSetMacro(RoomXMLFileName, std::string);

  /** Set detector displacement for LFOV*/
  itkGetMacro(DetectorOffset, float);
  itkSetMacro(DetectorOffset, float);


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
  ImagXGeometryReader():m_Geometry(NULL), m_DetectorOffset(0.f) {};

  ~ImagXGeometryReader() {};


private:
  //purposely not implemented
  ImagXGeometryReader(const Self&);
  void operator=(const Self&);

  virtual void GenerateData();

  GeometryType::Pointer m_Geometry;
  std::string           m_CalibrationXMLFileName;
  std::string           m_RoomXMLFileName;
  FileNamesContainer    m_ProjectionsFileNames;
  float                 m_DetectorOffset;
};

}
#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkImagXGeometryReader.txx"
#endif

#endif // __rtkImagXGeometryReader_h
