/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkElektaSynergyGeometryReader_h
#define rtkElektaSynergyGeometryReader_h

#include "RTKExport.h"
#include <itkLightProcessObject.h>
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

/** \class ElektaSynergyGeometryReader
 *
 * Creates a 3D circular geometry from the Elekta database input.
 *
 * \test rtkelektatest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK IOFilters
 */
class RTK_EXPORT ElektaSynergyGeometryReader : public itk::LightProcessObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ElektaSynergyGeometryReader);

  /** Standard type alias */
  using Self = ElektaSynergyGeometryReader;
  using Superclass = itk::LightProcessObject;
  using Pointer = itk::SmartPointer<Self>;

  /** Convenient type alias */
  using GeometryType = ThreeDCircularProjectionGeometry;

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ElektaSynergyGeometryReader);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Get the pointer to the generated geometry object. */
  itkGetMacro(Geometry, GeometryType::Pointer);

  /** Set the dicom unique ID of the acquisition, usually contained in the
   * name of the directory containing the projection images. The name is
   * of the form img_DicomUID. */
  itkGetMacro(DicomUID, std::string);
  itkSetMacro(DicomUID, std::string);

  /** Set the path to the file IMAGE.DBF */
  itkGetMacro(ImageDbfFileName, std::string);
  itkSetMacro(ImageDbfFileName, std::string);

  /** Set the path to the file FRAME.DBF */
  itkGetMacro(FrameDbfFileName, std::string);
  itkSetMacro(FrameDbfFileName, std::string);

protected:
  ElektaSynergyGeometryReader();


private:
  std::string
  GetImageIDFromDicomUID();
  void
  GetProjInfoFromDB(const std::string &  imageID,
                    std::vector<float> & projAngle,
                    std::vector<float> & projFlexX,
                    std::vector<float> & projFlexY);

  void
  GenerateData() override;

  GeometryType::Pointer m_Geometry;
  std::string           m_DicomUID;
  std::string           m_ImageDbfFileName;
  std::string           m_FrameDbfFileName;
};

} // namespace rtk
#endif
