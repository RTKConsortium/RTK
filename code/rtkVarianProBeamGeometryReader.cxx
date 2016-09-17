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


#include "rtkVarianProBeamGeometryReader.h"
#include "rtkVarianProBeamXMLFileReader.h"
#include "rtkXimImageIOFactory.h"

#include <itkImageFileReader.h>
#include <itksys/SystemTools.hxx>

rtk::VarianProBeamGeometryReader
::VarianProBeamGeometryReader():
  m_Geometry(NULL)
{
}

void
rtk::VarianProBeamGeometryReader
::GenerateData()
{
  // Create new RTK geometry object
  m_Geometry = GeometryType::New();

  // Read Varian XML file (for common geometric information)
  rtk::VarianProBeamXMLFileReader::Pointer proBeamXmlReader;
  proBeamXmlReader = rtk::VarianProBeamXMLFileReader::New();
  proBeamXmlReader->SetFilename(m_XMLFileName);
  proBeamXmlReader->GenerateOutputInformation();

  // Constants used to generate projection matrices
  itk::MetaDataDictionary &dic = *(proBeamXmlReader->GetOutputObject() );
  typedef itk::MetaDataObject< double > MetaDataDoubleType;
  const double sdd = dynamic_cast<MetaDataDoubleType *>(dic["SID"].GetPointer() )->GetMetaDataObjectValue();
  const double sid = dynamic_cast<MetaDataDoubleType *>(dic["SAD"].GetPointer() )->GetMetaDataObjectValue();
  typedef itk::MetaDataObject< std::string > MetaDataStringType;
  /*
  double offsetx;
  std::string fanType = dynamic_cast<const MetaDataStringType *>(dic["Fan"].GetPointer() )->GetMetaDataObjectValue();
  if(itksys::SystemTools::Strucmp(fanType.c_str(), "HALF") == 0)
    {
    // Half Fan (offset detector), get lateral offset from XML file
    offsetx = // Kept in case XML is different for HalfFan hardware.
      //dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetX"].GetPointer() )->GetMetaDataObjectValue() +
      dynamic_cast<MetaDataDoubleType *>(dic["ImagerLat"].GetPointer() )->GetMetaDataObjectValue();
    }
  else
    {
    // Full Fan (centered detector)
	offsetx = 0.0; //XML file doesn't contain offset!!
      //dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetX"].GetPointer() )->GetMetaDataObjectValue();
    }
  const double offsety = 0.0; //XML file doesn't contain offset!!
    //dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetY"].GetPointer() )->GetMetaDataObjectValue();
  */
  
  // Projections reader (for angle)
  rtk::XimImageIOFactory::RegisterOneFactory();
  // Projection matrices
  for(unsigned int noProj=0; noProj<m_ProjectionsFileNames.size(); noProj++)
    {
    typedef unsigned int                    InputPixelType;
    typedef itk::Image< InputPixelType, 2 > InputImageType;

    typedef itk::ImageFileReader< InputImageType > ReaderType;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( m_ProjectionsFileNames[noProj] );
    reader->UpdateOutputInformation();

	const double angle = 
	  dynamic_cast<MetaDataDoubleType *>(reader->GetMetaDataDictionary()["dCTProjectionAngle"].GetPointer())->GetMetaDataObjectValue();
	
	if (angle != 6000){
	  const double offsetx =
		dynamic_cast<MetaDataDoubleType *>(reader->GetMetaDataDictionary()["dDetectorOffsetX"].GetPointer())->GetMetaDataObjectValue();
	  const double offsety =
		dynamic_cast<MetaDataDoubleType *>(reader->GetMetaDataDictionary()["dDetectorOffsetY"].GetPointer())->GetMetaDataObjectValue();
	  m_Geometry->AddProjection(sid, sdd, angle, offsetx, offsety);
	  }
    }
}
