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


#include "rtkVarianProBeamGeometryReader.h"
#include "rtkVarianProBeamXMLFileReader.h"
#include "rtkXimImageIOFactory.h"

#include <itkImageFileReader.h>
#include <itkMacro.h>
#include <itkMetaDataObject.h>
#include <itksys/SystemTools.hxx>

rtk::VarianProBeamGeometryReader ::VarianProBeamGeometryReader()
  : m_Geometry(nullptr)
{}

void
rtk::VarianProBeamGeometryReader ::GenerateData()
{
  // Create new RTK geometry object
  m_Geometry = GeometryType::New();

  // Read Varian XML file (for common geometric information)
  rtk::VarianProBeamXMLFileReader::Pointer proBeamXmlReader;
  proBeamXmlReader = rtk::VarianProBeamXMLFileReader::New();
  proBeamXmlReader->SetFilename(m_XMLFileName);
  proBeamXmlReader->GenerateOutputInformation();

  // Constants used to generate projection matrices
  itk::MetaDataDictionary & dic = *(proBeamXmlReader->GetOutputObject());
  using MetaDataDoubleType = itk::MetaDataObject<double>;
  auto * sddMetaData = dynamic_cast<MetaDataDoubleType *>(dic["SID"].GetPointer());
  if (sddMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"SID\".");
  const double sdd = sddMetaData->GetMetaDataObjectValue();

  auto * sidMetaData = dynamic_cast<MetaDataDoubleType *>(dic["SAD"].GetPointer());
  if (sidMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"SAD\".");
  const double sid = sidMetaData->GetMetaDataObjectValue();

  // Projections reader (for angle)
  rtk::XimImageIOFactory::RegisterOneFactory();
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
    if (angle != 6000)
    {
      /* Warning: The offsets in the test scans were very small,
      however this configuration improved reconstruction quality slightly.*/
      auto * offsetXMetaData = dynamic_cast<MetaDataDoubleType *>(projectionDic["dDetectorOffsetX"].GetPointer());
      if (offsetXMetaData == nullptr)
        itkExceptionMacro(<< "Missing or invalid metadata \"dDetectorOffsetX\" in projection " << projectionsFileName
                          << ".");
      const double offsetx = offsetXMetaData->GetMetaDataObjectValue();

      auto * offsetYMetaData = dynamic_cast<MetaDataDoubleType *>(projectionDic["dDetectorOffsetY"].GetPointer());
      if (offsetYMetaData == nullptr)
        itkExceptionMacro(<< "Missing or invalid metadata \"dDetectorOffsetY\" in projection " << projectionsFileName
                          << ".");
      const double offsety = offsetYMetaData->GetMetaDataObjectValue();
      /*The angle-direction of RTK is opposite of the Xim properties
      (There doesn't seem to be a flag for direction in neither the xml nor xim file) */
      m_Geometry->AddProjection(sid, sdd, 180.0 - angle, offsetx, offsety);
    }
  }
}
