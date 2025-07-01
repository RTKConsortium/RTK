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
  const double sdd = dynamic_cast<MetaDataDoubleType *>(dic["CalibratedSID"].GetPointer())->GetMetaDataObjectValue();
  const double sid = dynamic_cast<MetaDataDoubleType *>(dic["CalibratedSAD"].GetPointer())->GetMetaDataObjectValue();

  using MetaDataStringType = itk::MetaDataObject<std::string>;
  double      offsetx = NAN;
  std::string fanType = dynamic_cast<const MetaDataStringType *>(dic["FanType"].GetPointer())->GetMetaDataObjectValue();
  if (itksys::SystemTools::Strucmp(fanType.c_str(), "HalfFan") == 0)
  {
    // Half Fan (offset detector), get lateral offset from XML file
    offsetx =
      dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetX"].GetPointer())->GetMetaDataObjectValue() +
      dynamic_cast<MetaDataDoubleType *>(dic["DetectorPosLat"].GetPointer())->GetMetaDataObjectValue();
  }
  else
  {
    // Full Fan (centered detector)
    offsetx =
      dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetX"].GetPointer())->GetMetaDataObjectValue();
  }
  const double offsety =
    dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetY"].GetPointer())->GetMetaDataObjectValue();

  // Projections reader (for angle)
  rtk::HndImageIOFactory::RegisterOneFactory();
  rtk::HncImageIOFactory::RegisterOneFactory();

  // Projection matrices
  for (const std::string & projectionsFileName : m_ProjectionsFileNames)
  {
    using InputPixelType = unsigned int;
    using InputImageType = itk::Image<InputPixelType, 2>;

    using ReaderType = itk::ImageFileReader<InputImageType>;
    auto reader = ReaderType::New();
    reader->SetFileName(projectionsFileName);
    reader->UpdateOutputInformation();

    const double angle =
      dynamic_cast<MetaDataDoubleType *>(reader->GetMetaDataDictionary()["dCTProjectionAngle"].GetPointer())
        ->GetMetaDataObjectValue();

    m_Geometry->AddProjection(sid, sdd, angle, offsetx, offsety);
  }
}
