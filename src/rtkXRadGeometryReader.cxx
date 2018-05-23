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


#include "rtkXRadGeometryReader.h"
#include "rtkXRadImageIOFactory.h"

#include <itkImageIOFactory.h>
#include <itkMetaDataObject.h>

rtk::XRadGeometryReader
::XRadGeometryReader():
  m_Geometry(ITK_NULLPTR)
{
}

void
rtk::XRadGeometryReader
::GenerateData()
{
  // Create new RTK geometry object
  m_Geometry = GeometryType::New();
  GeometryType::Pointer tmpGeo = GeometryType::New();

  // Read image information which contains geometry information
  rtk::XRadImageIOFactory::RegisterOneFactory();
  itk::ImageIOBase::Pointer reader =
        itk::ImageIOFactory::CreateImageIO(m_ImageFileName.c_str(), itk::ImageIOFactory::ReadMode);
  if(!reader)
    itkExceptionMacro(<< m_ImageFileName << " is not an XRad file.");
  reader->SetFileName(m_ImageFileName);
  reader->ReadImageInformation();

  std::string sectionName, paramName;
  typedef itk::MetaDataObject< std::string > MetaDataStringType;
  itk::MetaDataDictionary &dic = reader->GetMetaDataDictionary();

  for(unsigned int i=0; i<reader->GetDimensions(2); i++)
    {
    std::ostringstream os;
    os << "iView3D(Projection " << i << ")";
    sectionName = os.str();

    paramName = sectionName + "_CBCT.ProjectionGeometryArray.u_axis";
    std::istringstream isu(dynamic_cast<MetaDataStringType*>(dic[paramName].GetPointer())->GetMetaDataObjectValue());
    paramName = sectionName + "_CBCT.ProjectionGeometryArray.v_axis";
    std::istringstream isv(dynamic_cast<MetaDataStringType*>(dic[paramName].GetPointer())->GetMetaDataObjectValue());
    paramName = sectionName + "_CBCT.ProjectionGeometryArray.focus";
    std::istringstream isfocus(dynamic_cast<MetaDataStringType*>(dic[paramName].GetPointer())->GetMetaDataObjectValue());
    paramName = sectionName + "_CBCT.ProjectionGeometryArray.center";
    std::istringstream iscenter(dynamic_cast<MetaDataStringType*>(dic[paramName].GetPointer())->GetMetaDataObjectValue());
    itk::Vector<double,3> u,v;
    itk::Vector<double,3> focus,center;
    for(unsigned int j=0; j<3; j++)
      {
      isu >> u[j];
      isv >> v[j];
      isfocus >> focus[j];
      iscenter >> center[j];

      // cm to mm
      focus[j] *= 10.;
      center[j] *= 10.;
      }

    // Change of coordinate system to IEC
    // The coordinate system of XRad is supposed to be illustrated in figure 1 of
    // [Clarkson et al, Med Phys, 2011]. Real acquisition of a Playmobil have been
    // actually used to be sure...
    u[0] *= -1.;
    v[0] *= -1.;
    focus[0] *= -1.;
    center[0] *= -1;
    u[2] *= -1.;
    v[2] *= -1.;
    focus[2] *= -1.;
    center[2] *= -1;
    std::swap(u[1], u[2]);
    std::swap(v[1], v[2]);
    std::swap(focus[1], focus[2]);
    std::swap(center[1], center[2]);

    u.Normalize();
    v.Normalize();
    tmpGeo->AddReg23Projection(&(focus[0]),
                               &(center[0]),
                               u, v);

    paramName = sectionName + "_CBCT.ProjectionGeometryArray.u_off";
    std::string suoff = dynamic_cast<MetaDataStringType*>(dic[paramName].GetPointer())->GetMetaDataObjectValue();
    double uoff = atof(suoff.c_str()) * reader->GetSpacing(0);

    paramName = sectionName + "_CBCT.ProjectionGeometryArray.v_off";
    std::string svoff = dynamic_cast<MetaDataStringType*>(dic[paramName].GetPointer())->GetMetaDataObjectValue();
    double voff = atof(svoff.c_str()) * reader->GetSpacing(1);

    m_Geometry->AddProjectionInRadians(tmpGeo->GetSourceToIsocenterDistances()[i],
                                       tmpGeo->GetSourceToDetectorDistances()[i],
                                       tmpGeo->GetGantryAngles()[i],
                                       tmpGeo->GetProjectionOffsetsX()[i]-uoff,
                                       tmpGeo->GetProjectionOffsetsY()[i]-voff,
                                       tmpGeo->GetOutOfPlaneAngles()[i],
                                       tmpGeo->GetInPlaneAngles()[i],
                                       tmpGeo->GetSourceOffsetsX()[i],
                                       tmpGeo->GetSourceOffsetsY()[i]);
    }
}
