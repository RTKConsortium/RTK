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


#include "math.h"

#include "rtkVarianObiGeometryReader.h"
#include "rtkVarianObiXMLFileReader.h"
#include "rtkHndImageIOFactory.h"
#include "rtkHncImageIOFactory.h"

#include <itkImageFileReader.h>
#include <itkMacro.h>
#include <itkMetaDataObject.h>
#include <itksys/SystemTools.hxx>

rtk::VarianObiGeometryReader ::VarianObiGeometryReader()
  : m_Geometry(nullptr)
{}

void
rtk::VarianObiGeometryReader ::GenerateData()
{
  // Create new RTK geometry object
  m_Geometry = GeometryType::New();

  // Read Varian XML file (for common geometric information)
  rtk::VarianObiXMLFileReader::Pointer obiXmlReader;
  obiXmlReader = rtk::VarianObiXMLFileReader::New();
  obiXmlReader->SetFilename(m_XMLFileName);
  obiXmlReader->GenerateOutputInformation();

  // Constants used to generate projection matrices
  itk::MetaDataDictionary & dic = *(obiXmlReader->GetOutputObject());
  using MetaDataDoubleType = itk::MetaDataObject<double>;
  auto * sddMetaData = dynamic_cast<MetaDataDoubleType *>(dic["CalibratedSID"].GetPointer());
  if (sddMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"CalibratedSID\".");
  const double sdd = sddMetaData->GetMetaDataObjectValue();

  auto * sidMetaData = dynamic_cast<MetaDataDoubleType *>(dic["CalibratedSAD"].GetPointer());
  if (sidMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"CalibratedSAD\".");
  const double sid = sidMetaData->GetMetaDataObjectValue();

  double offsetx = NAN;
  auto * fanTypeMetaData = dynamic_cast<const itk::MetaDataObject<std::string> *>(dic["FanType"].GetPointer());
  if (fanTypeMetaData == nullptr)
  {
    itkExceptionMacro(<< "Missing or invalid metadata \"FanType\".");
  }
  std::string fanType = fanTypeMetaData->GetMetaDataObjectValue();
  if (itksys::SystemTools::Strucmp(fanType.c_str(), "HalfFan") == 0)
  {
    // Half Fan (offset detector), get lateral offset from XML file
    auto * calibratedDetectorOffsetXMetaData =
      dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetX"].GetPointer());
    if (calibratedDetectorOffsetXMetaData == nullptr)
      itkExceptionMacro(<< "Missing or invalid metadata \"CalibratedDetectorOffsetX\".");
    const double calibratedDetectorOffsetX = calibratedDetectorOffsetXMetaData->GetMetaDataObjectValue();

    auto * detectorPosLatMetaData = dynamic_cast<MetaDataDoubleType *>(dic["DetectorPosLat"].GetPointer());
    if (detectorPosLatMetaData == nullptr)
      itkExceptionMacro(<< "Missing or invalid metadata \"DetectorPosLat\".");
    const double detectorPosLat = detectorPosLatMetaData->GetMetaDataObjectValue();

    offsetx = calibratedDetectorOffsetX + detectorPosLat;
  }
  else
  {
    // Full Fan (centered detector)
    auto * calibratedDetectorOffsetXMetaData =
      dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetX"].GetPointer());
    if (calibratedDetectorOffsetXMetaData == nullptr)
      itkExceptionMacro(<< "Missing or invalid metadata \"CalibratedDetectorOffsetX\".");
    offsetx = calibratedDetectorOffsetXMetaData->GetMetaDataObjectValue();
  }
  auto * offsetYMetaData = dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetY"].GetPointer());
  if (offsetYMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"CalibratedDetectorOffsetY\".");
  const double offsety = offsetYMetaData->GetMetaDataObjectValue();

  // Projections reader (for angle)
  rtk::HndImageIOFactory::RegisterOneFactory();
  rtk::HncImageIOFactory::RegisterOneFactory();

  // Projection matrices
  for (const std::string & projectionsFileName : m_ProjectionsFileNames)
  {

    auto reader = itk::ImageFileReader<itk::Image<unsigned int, 2>>::New();
    reader->SetFileName(projectionsFileName);
    reader->UpdateOutputInformation();

    itk::MetaDataDictionary & projectionDic = reader->GetMetaDataDictionary();
    auto * angleMetaData = dynamic_cast<MetaDataDoubleType *>(projectionDic["dCTProjectionAngle"].GetPointer());
    if (angleMetaData == nullptr)
      itkExceptionMacro(<< "Missing or invalid metadata \"dCTProjectionAngle\" in projection " << projectionsFileName
                        << ".");
    const double angle = angleMetaData->GetMetaDataObjectValue();

    m_Geometry->AddProjection(sid, sdd, angle, offsetx, offsety);
  }
}
