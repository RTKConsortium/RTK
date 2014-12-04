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

#ifndef __rtkImagXGeometryReader_txx
#define __rtkImagXGeometryReader_txx

#include "rtkMacro.h"
#include "rtkImagXXMLFileReader.h"
#include <itkDOMNodeXMLReader.h>
#include <itkImageFileReader.h>
#include <itkGDCMImageIO.h>

namespace rtk
{

template< typename TInputImage >
void ImagXGeometryReader<TInputImage>::GenerateData()
{
  // Create new RTK geometry object
  m_Geometry = GeometryType::New();

  // Reading iMagX calibration file
  // iMagX axisName parameter
  std::string axisName, sid_s, sdd_s;
  // iMagX deformation model parameters
  std::vector<float> Px(5,0.f),Py(5,0.f),Pz(5,0.f), // Detector translations
                     Rx(5,0.f),Ry(5,0.f),Rz(5,0.f), // Detector rotations
                     Tx(5,0.f),Ty(5,0.f),Tz(5,0.f); // Source translations

  itk::DOMNode::Pointer XMLFile;
  itk::DOMNodeXMLReader::Pointer readerXML = itk::DOMNodeXMLReader::New();
  readerXML->SetFileName( m_CalibrationXMLFileName );
  readerXML->Update();
  XMLFile = readerXML->GetOutput();

  itk::DOMNode::AttributesListType list;
  itk::DOMNode::AttributesListType::const_iterator list_it;
  itk::DOMNode::ChildrenListType list_child;
  XMLFile->GetAllChildren(list_child);

  for(unsigned int i = 0; i< list_child.size(); i++)
  {
    list_child[i]->GetAllAttributes(list);
    unsigned int k = 0;
    for(list_it = list.begin(); list_it != list.end(); list_it++, k++)
    {
      if( (*list_it).first.c_str() == std::string("axis") )
        axisName = (*list_it).second.c_str();
    }

    // If cbct axis then extract deformation model parameters
    if( axisName == std::string("CBCT") )
    {
      itk::DOMNode::ChildrenListType list_child2;
      list_child[i]->GetAllChildren(list_child2);
      for(unsigned int l = 0; l<list_child2.size(); l++)
      {
        itk::DOMNode::ChildrenListType list_child3;
        list_child2[l]->GetAllChildren(list_child3);
        unsigned int p=0;
        for(unsigned int n = 0; n<list_child3.size(); n++, p++)
        {
          itk::DOMNode::AttributesListType list2;
          list_child3[n]->GetAllAttributes(list2);
          unsigned int m = 0;
          for(list_it = list2.begin(); list_it != list2.end(); list_it++, m++)
          {
            if( ( list_child3[n]->GetName() == std::string("Px") ) &&
                ( (*list_it).first.c_str() != std::string("MSE") ) )
              Px[m-1] = std::atof( (*list_it).second.c_str() );
            else if( ( list_child3[n]->GetName() == std::string("Py") ) &&
                     ( (*list_it).first.c_str() != std::string("MSE") ) )
              Py[m-1] = std::atof( (*list_it).second.c_str() );
            else if( ( list_child3[n]->GetName() == std::string("Pz") ) &&
                     ( (*list_it).first.c_str() != std::string("MSE") ) )
              Pz[m-1] = std::atof( (*list_it).second.c_str() );
            else if( ( list_child3[n]->GetName() == std::string("Rx") ) &&
                     ( (*list_it).first.c_str() != std::string("MSE") ) )
              Rx[m-1] = std::atof( (*list_it).second.c_str() );
            else if( ( list_child3[n]->GetName() == std::string("Ry") ) &&
                     ( (*list_it).first.c_str() != std::string("MSE") ) )
              Ry[m-1] = std::atof( (*list_it).second.c_str() );
            else if( ( list_child3[n]->GetName() == std::string("Rz") ) &&
                     ( (*list_it).first.c_str() != std::string("MSE") ) )
              Rz[m-1] = std::atof( (*list_it).second.c_str() );
            else if( ( list_child3[n]->GetName() == std::string("Tx") ) &&
                     ( (*list_it).first.c_str() != std::string("MSE") ) )
              Tx[m-1] = std::atof( (*list_it).second.c_str() );
            else if( ( list_child3[n]->GetName() == std::string("Ty") ) &&
                     ( (*list_it).first.c_str() != std::string("MSE") ) )
              Ty[m-1] = std::atof( (*list_it).second.c_str() );
            else if( ( list_child3[n]->GetName() == std::string("Tz") ) &&
                     ( (*list_it).first.c_str() != std::string("MSE") ) )
              Tz[m-1] = std::atof( (*list_it).second.c_str() );
          }
        }
      }
    }
  }
  list.clear();
  list_child.clear();

  // Reading iMagX room setup for SID and SDD
  readerXML->SetFileName( m_RoomXMLFileName );
  readerXML->Update();

  XMLFile = readerXML->GetOutput();
  XMLFile->GetAllChildren(list_child);

  for(unsigned int i = 0; i < list_child.size(); i++)
  {
    list_child[i]->GetAllAttributes(list);
    unsigned int k = 0;
    for(list_it = list.begin(); list_it != list.end(); list_it++, k++)
    {
      if( (*list_it).first.c_str() == std::string("axis") )
        axisName = (*list_it).second.c_str();
    }

    // If cbct axis then extract sdd and sid parameters
    if( axisName == std::string("CBCT") )
    {
      itk::DOMNode::ChildrenListType list_child2;
      list_child[i]->GetAllChildren(list_child2);
      for(unsigned int l = 0; l<list_child2.size(); l++)
      {
        itk::DOMNode::AttributesListType list2;
        list_child2[l]->GetAllAttributes(list2);
        unsigned int p = 0;
        for(list_it = list2.begin(); list_it != list2.end(); list_it++, k++)
        {
          if( (*list_it).first.c_str() == std::string("sdd") )
            sdd_s = (*list_it).second.c_str();
          if( (*list_it).first.c_str() == std::string("sid") )
            sid_s = (*list_it).second.c_str();
        }
      }
    }
  }
  // Deformation computation following model:
  //
  //    (a0 + a1*t) + a2*cos(a3*t + a4)
  //
  std::vector<float> detTrans(3,0.f), detRot(3,0.f), srcTrans(3,0.f);
  float gantryAngle    = 0.f;
  float gantryAngleRad = 0.f;
  float sid = std::atof(sid_s.c_str());
  float sdd = std::atof(sdd_s.c_str());
  float deg2rad = float(M_PI)/180.f;

  // Projection matrices
  for(unsigned int noProj=0; noProj < m_ProjectionsFileNames.size(); noProj++)
  {

    itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO( m_ProjectionsFileNames[0].c_str(), itk::ImageIOFactory::ReadMode );
    typedef itk::ImageFileReader< TInputImage > ReaderType;
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( m_ProjectionsFileNames[noProj] );
    imageIO = itk::GDCMImageIO::New();
    reader->SetImageIO(imageIO);
    reader->UpdateOutputInformation();

    // Reading Gantry Angle
    std::string tagkey = "300a|011e";
    std::string labelId, value;
    itk::GDCMImageIO::GetLabelFromTag( tagkey, labelId );
    dynamic_cast<itk::GDCMImageIO*>(imageIO.GetPointer())->GetValueFromTag(tagkey, value);

    gantryAngle = std::atof(value.c_str());
    gantryAngleRad = gantryAngle*deg2rad;
    // Detector translation deformations
    detTrans[0] = Px[0] + Px[1]*gantryAngle + Px[2]*vcl_cos(Px[3]*gantryAngleRad + Px[4])
                + m_DetectorOffset;
    detTrans[1] = Py[0] + Py[1]*gantryAngle + Py[2]*vcl_cos(Py[3]*gantryAngleRad + Py[4]);
    detTrans[2] = Pz[0] + Pz[1]*gantryAngle + Pz[2]*vcl_cos(Pz[3]*gantryAngleRad + Pz[4]);
    // Detector rotation deformations
    detRot[0] = Rx[0] + Rx[1]*gantryAngle + Rx[2]*vcl_cos(Rx[3]*gantryAngleRad + Rx[4]);
    detRot[1] = Ry[0] + Ry[1]*gantryAngle + Ry[2]*vcl_cos(Ry[3]*gantryAngleRad + Ry[4]);
    detRot[2] = Rz[0] + Rz[1]*gantryAngle + Rz[2]*vcl_cos(Rz[3]*gantryAngleRad + Rz[4]);
    // Source translation deformations
    srcTrans[0] = Tx[0] + Tx[1]*gantryAngle + Tx[2]*vcl_cos(Tx[3]*gantryAngleRad + Tx[4]);
    srcTrans[1] = Ty[0] + Ty[1]*gantryAngle + Ty[2]*vcl_cos(Ty[3]*gantryAngleRad + Ty[4]);
    srcTrans[2] = Tz[0] + Tz[1]*gantryAngle + Tz[2]*vcl_cos(Tz[3]*gantryAngleRad + Tz[4]);

    // Conversion from iMagX geometry to RTK geometry standard
    float aid  = sid-sdd;
    float sidp = srcTrans[2] + sid;
    float Pzp  = detTrans[2] + aid;
    float bx   = detRot[0]*deg2rad;
    float by   = detRot[1]*deg2rad;
    float bz   = detRot[2]*deg2rad;

    float gAngle          = gantryAngle + detRot[1];
    float outOfPlaneAngle = detRot[0];
    float inPlaneAngle    = detRot[2];
    float sourceToIsocenterDistance = cos(bx)*(cos(by)*sidp+sin(by)*srcTrans[0])-sin(bx)*srcTrans[1];
    float sourceToDetectorDistance  = sourceToIsocenterDistance-cos(bx)*(cos(by)*Pzp+sin(by)*detTrans[0])+sin(bx*detTrans[1]);

    float fac11 =   cos(by)*sidp  + sin(by)*srcTrans[0];
    float fac12 =   sin(bx)*fac11 + cos(bx)*srcTrans[1];
    float fac13 = - sin(by)*sidp  + cos(by)*srcTrans[0];
    float sourceOffsetX = sin(bz)*fac12 + cos(bz)*fac13;
    float sourceOffsetY = cos(bz)*fac12 - sin(bz)*fac13;

    float fac21 =   cos(by)*Pzp   + sin(by)*detTrans[0];
    float fac22 =   sin(bx)*fac21 + cos(bx)*detTrans[1];
    float fac23 = - sin(by)*Pzp   + cos(by)*detTrans[0];
    float projectionOffsetX = sin(bz)*fac22 + cos(bz)*fac23;
    float projectionOffsetY = cos(bz)*fac22 - sin(bz)*fac23;

    m_Geometry->AddProjection(sourceToIsocenterDistance,
                              sourceToDetectorDistance,
                              gAngle,
                              projectionOffsetX,
                              projectionOffsetY,
                              outOfPlaneAngle,
                              inPlaneAngle,
                              sourceOffsetX,
                              sourceOffsetY);
  }
}
} //namespace rtk
#endif
