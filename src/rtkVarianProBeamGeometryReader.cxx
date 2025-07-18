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
  const double sdd = dynamic_cast<MetaDataDoubleType *>(dic["SID"].GetPointer())->GetMetaDataObjectValue();
  const double sid = dynamic_cast<MetaDataDoubleType *>(dic["SAD"].GetPointer())->GetMetaDataObjectValue();

  // Projections reader (for angle)
  rtk::XimImageIOFactory::RegisterOneFactory();
  // Projection matrices
  for (const std::string & projectionsFileName : m_ProjectionsFileNames)
  {

    auto reader = itk::ImageFileReader<itk::Image<unsigned int, 2>>::New();
    reader->SetFileName(projectionsFileName);
    reader->UpdateOutputInformation();

    const double angle =
      dynamic_cast<MetaDataDoubleType *>(reader->GetMetaDataDictionary()["dCTProjectionAngle"].GetPointer())
        ->GetMetaDataObjectValue();
    if (angle != 6000)
    {
      /* Warning: The offsets in the test scans were very small,
      however this configuration improved reconstruction quality slightly.*/
      const double offsetx =
        dynamic_cast<MetaDataDoubleType *>(reader->GetMetaDataDictionary()["dDetectorOffsetX"].GetPointer())
          ->GetMetaDataObjectValue();
      const double offsety =
        dynamic_cast<MetaDataDoubleType *>(reader->GetMetaDataDictionary()["dDetectorOffsetY"].GetPointer())
          ->GetMetaDataObjectValue();
      /*The angle-direction of RTK is opposite of the Xim properties
      (There doesn't seem to be a flag for direction in neither the xml nor xim file) */
      m_Geometry->AddProjection(sid, sdd, 180.0 - angle, offsetx, offsety);
    }
  }
}
