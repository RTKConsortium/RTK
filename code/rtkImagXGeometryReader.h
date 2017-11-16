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

#ifndef rtkImagXGeometryReader_h
#define rtkImagXGeometryReader_h

#include <itkLightProcessObject.h>
#include "rtkThreeDCircularProjectionGeometry.h"

#include <vector>

namespace rtk
{

/** \class ImagXGeometryReader
 *
 * Creates a 3D circular geometry from the IBA data set.
 *
 * \test rtkimagxtest.cxx
 *
 * \author Marc Vila, C. Mory, S. Brousmiche (IBA)
 *
 * \ingroup IOFilters
 */
template< typename TInputImage >
class ImagXGeometryReader : public itk::LightProcessObject
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

  /** Set the iMagX calibration xml file
    */
  itkGetMacro(CalibrationXMLFileName, std::string);
  itkSetMacro(CalibrationXMLFileName, std::string);

  /** Set the iMagX room setup xml file*/
  itkGetMacro(RoomXMLFileName, std::string);
  itkSetMacro(RoomXMLFileName, std::string);

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
  ImagXGeometryReader()
     : m_Geometry(ITK_NULLPTR), m_CalibrationXMLFileName(""), m_RoomXMLFileName("") {};

  ~ImagXGeometryReader() {}


private:
  //purposely not implemented
  ImagXGeometryReader(const Self&);
  void operator=(const Self&);

  void GenerateData() ITK_OVERRIDE;

  // DICOM tag for AI versions
  static const std::string m_AI_VERSION_1p2;
  static const std::string m_AI_VERSION_1p5;
  static const std::string m_AI_VERSION_2p1;

  // DICOM tags depend on AI version
  std::string getAIversion();

  // Structure containing the flexmap (for AI versions >= 2.0)
  struct FlexmapType {
      bool isValid;
      std::string activeArcName;
      std::string activeGeocalUID;
      float sid, sdd, sourceToNozzleOffsetAngle;
      float constantDetectorOffset, xMinus, xPlus;
      bool isCW;
      std::vector<float> anglesDeg;  // Gantry angles [deg]
      std::vector<float> Px, Py, Pz, // Detector translations
                         Rx, Ry, Rz, // Detector rotations
                         Tx, Ty, Tz; // Source translations
  };

  FlexmapType GetGeometryForAI2p1();

  struct InterpResultType {
      int id0, id1; // indices of angles just below and above the target angle
      float a0, a1; // weights (1/distance) to angles below and above
  };

  InterpResultType interpolate(const std::vector<float>& flexAngles, bool isCW, float angleDegree);
  
  // Structure containing the calibration models (for AI versions < 2.0)
  struct CalibrationModelType {
      bool isValid;
      float sid, sdd, sourceToNozzleOffsetAngle;
      std::vector<float> Px, Py, Pz, // Detector translations model
                         Rx, Ry, Rz, // Detector rotations model
                         Tx, Ty, Tz; // Source translations model

      CalibrationModelType() {
          sourceToNozzleOffsetAngle = -90.f; 
          Px = std::vector<float>(5, 0.f);
          Py = std::vector<float>(5, 0.f);
          Pz = std::vector<float>(5, 0.f);
          Rx = std::vector<float>(5, 0.f);
          Ry = std::vector<float>(5, 0.f);
          Rz = std::vector<float>(5, 0.f);
          Tx = std::vector<float>(5, 0.f);
          Ty = std::vector<float>(5, 0.f);
          Tz = std::vector<float>(5, 0.f);
      }
  };
  
  CalibrationModelType GetGeometryForAI1p5();

  CalibrationModelType GetGeometryForAI1p5FromXMLFiles();

  bool isCW(const std::vector<float>& angles);

  std::vector<float> getInterpolatedValue(const InterpResultType& ires, const std::vector<float>& Dx, const std::vector<float>& Dy, const std::vector<float>& Dz);

  // Evaluate the calibration models for a given angle
  std::vector<float> getDeformations(float gantryAngle, const std::vector<float>& Dx, const std::vector<float>& Dy, const std::vector<float>& Dz);
  
  void addEntryToGeometry(float gantryAngleDegree, float nozzleToRadAngleOffset, float sid, float sdd, 
                          std::vector<float>& P, std::vector<float>& R, std::vector<float>& T);

  void addEntryToGeometry(const FlexmapType& flex, float gantryAngleDegree);

  void addEntryToGeometry(const CalibrationModelType& calibModel, float gantryAngleDegree);

  GeometryType::Pointer m_Geometry;
  std::string           m_CalibrationXMLFileName;
  std::string           m_RoomXMLFileName;
  FileNamesContainer    m_ProjectionsFileNames;
};

}
#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkImagXGeometryReader.hxx"
#endif

#endif // rtkImagXGeometryReader_h
