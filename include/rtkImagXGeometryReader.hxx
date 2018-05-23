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

#ifndef rtkImagXGeometryReader_hxx
#define rtkImagXGeometryReader_hxx

#include "rtkImagXGeometryReader.h"

#include "rtkMacro.h"
#include "rtkImagXXMLFileReader.h"
#include <itkDOMNodeXMLReader.h>
#include <itkImageFileReader.h>
#include <itkGDCMImageIO.h>

#include <string>
#include <istream>
#include <iterator>

namespace rtk
{
template< typename TInputImage >
const std::string ImagXGeometryReader<TInputImage>::m_AI_VERSION_1p2 = "IMAGX:1.2";

template< typename TInputImage >
const std::string ImagXGeometryReader<TInputImage>::m_AI_VERSION_1p5 = "IMAGX:1.6";

template< typename TInputImage >
const std::string ImagXGeometryReader<TInputImage>::m_AI_VERSION_2pX = "adaPTinsight: 2";

template< typename TInputImage >
typename ImagXGeometryReader<TInputImage>::FlexmapType
ImagXGeometryReader<TInputImage>::GetGeometryForAI2p1()
{
  FlexmapType F;

  F.isValid = false;

  // Read the dcmdump of the geocal tag (3001,1018)

  itk::DOMNodeXMLReader::Pointer readerXML = itk::DOMNodeXMLReader::New();
  readerXML->SetFileName(m_CalibrationXMLFileName);
  readerXML->Update();
  itk::DOMNode::Pointer XMLFile = readerXML->GetOutput();

  itk::DOMNode::AttributesListType::const_iterator list_it;
  itk::DOMNode::ChildrenListType list_child;
  XMLFile->GetAllChildren(list_child);

  bool arcFound = false;
  bool FPOffsetfFound = false;
  bool gantryParametersFound = false;
  itk::DOMNode::AttributesListType list;
  for (unsigned int i = 0; i < list_child.size(); i++)
    {
    std::string tagName = list_child[i]->GetName();

    // 1. From all available arcs, find the active one
    if (!tagName.compare("arc") && !arcFound)
      {
      list.clear();
      list_child[i]->GetAllAttributes(list);
      for (list_it = list.begin(); list_it != list.end(); list_it++)
        {
        std::string subTagName = (*list_it).first.c_str();
        std::string tagValue = (*list_it).second.c_str();
        if (!subTagName.compare("name") ) {
          F.activeArcName = tagValue;
          }
        else if (!subTagName.compare("geoCalModelUid") )
          {
          F.activeGeocalUID = tagValue;
          }
        else if (!subTagName.compare("active") && tagValue == "true")
          {
          arcFound = true;
          }
        }
      }

    // 2. Check detector offset values in MINUS/PLUS positions
    if (!tagName.compare("FPOffsets") )
      {
      bool xPlusFound = false;
      bool xMinusFound = false;
      list.clear();
      list_child[i]->GetAllAttributes(list);
      for (list_it = list.begin(); list_it != list.end(); list_it++)
        {
        std::string subTagName = (*list_it).first.c_str();
        std::string tagValue = (*list_it).second.c_str();
        if (!subTagName.compare("xPlus") ) {
          F.xPlus = std::atof(tagValue.c_str() );
          xPlusFound = true;
          }
        else if (!subTagName.compare("xMinus") )
          {
          F.xMinus = std::atof(tagValue.c_str() );
          xMinusFound = true;
          }
        }

      FPOffsetfFound = xPlusFound & xMinusFound;
      }

    // 3. Get gantry parameters
    if (!tagName.compare("gantryParameters") )
      {
      bool sidFound = false;
      bool sadFound = false;
      bool angleOffsetFound = false;
      list.clear();
      list_child[i]->GetAllAttributes(list);
      for (list_it = list.begin(); list_it != list.end(); list_it++)
        {
        std::string subTagName = (*list_it).first.c_str();
        std::string subTagValue = (*list_it).second.c_str();
        if (!subTagName.compare("sid") )
          {
          F.sdd = std::atof(subTagValue.c_str() );
          sidFound = true;
          }
        else if (!subTagName.compare("sad") )
          {
          F.sid = std::atof(subTagValue.c_str() );
          sadFound = true;
          }
        else if (!subTagName.compare("sourceToNozzleOffsetAngle") )
          {
          F.sourceToNozzleOffsetAngle = std::atof(subTagValue.c_str() );
          angleOffsetFound = true;
          }
        }
      gantryParametersFound = sidFound & sadFound & angleOffsetFound;
      }
    }

  // Depending on arc found
  bool flexmapFoundAndLoaded = false;
  if (arcFound)
    {
    std::cout << "Found active arc '" << F.activeArcName << "' linked to geocal UID '" << F.activeGeocalUID << "'" <<
      std::endl;

    // Parse tag and check if 'PLUS' or 'MINUS'
    if (F.activeArcName.find("PLUS") != std::string::npos) {
      F.constantDetectorOffset = F.xPlus;
      }
    else if (F.activeArcName.find("MINUS") != std::string::npos) {
      F.constantDetectorOffset = F.xMinus;
      }

    // Get flexmap
    for (unsigned int i = 0; i < list_child.size(); i++)
      {
      if (list_child[i]->GetName() == "geometricalCalibrationModels")
        {
        itk::DOMNode::ChildrenListType list_geocals;
        list_child[i]->GetAllChildren(list_geocals);

        for (unsigned int gid = 0; gid < list_geocals.size(); gid++)
          {
          if (list_geocals[gid]->GetName() == "geoCalModel")
            {
            list.clear();
            list_geocals[gid]->GetAllAttributes(list);
            for (list_it = list.begin(); list_it != list.end(); list_it++)
              {
              std::string tagName = (*list_it).first.c_str();
              std::string tagValue = (*list_it).second.c_str();
              if (tagName == "uid" && tagValue == F.activeGeocalUID) {
                flexmapFoundAndLoaded = true;
                }
              }
            }

          if (flexmapFoundAndLoaded)
            {
            itk::DOMNode::ChildrenListType list_flexmap;
            list_geocals[gid]->GetAllChildren(list_flexmap);

            for (unsigned int flist_id = 0; flist_id < list_flexmap.size(); flist_id++)
              {
              std::string str = dynamic_cast<itk::DOMTextNode*>(list_flexmap[flist_id])->GetText();
              std::stringstream iss(str);
              std::vector<float> v;
              while (iss.good() )
                {
                std::string substr;
                std::getline(iss, substr, ',');
                v.push_back(std::atof(substr.c_str() ) );
                }

              if (v.size() == 10) {
                F.anglesDeg.push_back(v[0]);
                F.Tx.push_back(v[1]);
                F.Ty.push_back(v[2]);
                F.Tz.push_back(v[3]);
                F.Px.push_back(v[4] + F.constantDetectorOffset);
                F.Py.push_back(v[5]);
                F.Pz.push_back(v[6]);
                F.Rx.push_back(v[7]);
                F.Ry.push_back(v[8]);
                F.Rz.push_back(v[9]);
                }
              }
            flexmapFoundAndLoaded = false; // One flexmap already loaded
            }

          if (F.anglesDeg.size() == 0) {
            flexmapFoundAndLoaded = false;
            }
          }
        }
      }
    }

  F.isCW = this->isCW(F.anglesDeg); // Needed for flexmap sampling
  F.isValid = arcFound & FPOffsetfFound & gantryParametersFound & flexmapFoundAndLoaded;

  return F;
}

template< typename TInputImage >
bool ImagXGeometryReader<TInputImage>::isCW(const std::vector<float>& angles)
{
  std::vector<float> cp;
  std::copy(angles.begin(), angles.end(), std::back_inserter(cp) );
  std::nth_element(cp.begin(), cp.begin() + cp.size() / 2, cp.end() );

  return (cp[cp.size() / 2] >= 0.f) ? true : false;
}

template< typename TInputImage >
typename ImagXGeometryReader<TInputImage>::CalibrationModelType
ImagXGeometryReader<TInputImage>::GetGeometryForAI1p5()
{
  CalibrationModelType Cm;

  Cm.isValid = false;

  // Create and set ImageIO
  itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(
      m_ProjectionsFileNames[0].c_str(), itk::ImageIOFactory::ReadMode);
  imageIO = itk::GDCMImageIO::New();
  dynamic_cast<itk::GDCMImageIO*>(imageIO.GetPointer() )->LoadPrivateTagsOn();

  // Define, create and set projection reader
  typedef itk::ImageFileReader< TInputImage > ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetImageIO(imageIO);
  reader->SetFileName(m_ProjectionsFileNames[0]);
  reader->UpdateOutputInformation();

  // Read room setup parameters in the DICOM info of the first projection
  std::string roomSetupTagKey = "3001|0012";
  std::string roomSetupInfo;
  dynamic_cast<itk::GDCMImageIO*>(imageIO.GetPointer() )->GetValueFromTag(roomSetupTagKey, roomSetupInfo);

  // Extract SID, SDD and angle offset from the roomSetupInfo
  itk::DOMNodeXMLReader::Pointer parser = itk::DOMNodeXMLReader::New();
  std::istringstream is(roomSetupInfo);
  parser->Update(is);

  Cm.sid = atof(parser->GetOutput()->GetChild("Axis")->GetChild("Distances")->GetAttribute("sid").c_str() );
  Cm.sdd = atof(parser->GetOutput()->GetChild("Axis")->GetChild("Distances")->GetAttribute("sdd").c_str() );
  Cm.sourceToNozzleOffsetAngle =
    atof(parser->GetOutput()->GetChild("Axis")->GetChild("AngleOffset")->GetAttribute("projection").c_str() );

  // Read calibration model's parameters in the DICOM info of the first projection
  std::string calibrationTagKey = "3001|0013";
  std::string calibrationInfo;
  dynamic_cast<itk::GDCMImageIO*>(imageIO.GetPointer() )->GetValueFromTag(calibrationTagKey, calibrationInfo);

  // Extract calibration model's parameters from calibrationInfo
  is.clear();
  is.str(calibrationInfo);
  parser->Update(is);

  itk::DOMNode::AttributesListType listPx, listPy, listPz, listRx, listRy, listRz, listTx, listTy, listTz;
  parser->GetOutput()->GetChild("Axis")->GetChild("DetectorTranslation")->GetChild("Px")->GetAllAttributes(listPx,
    true);
  parser->GetOutput()->GetChild("Axis")->GetChild("DetectorTranslation")->GetChild("Py")->GetAllAttributes(listPy,
    true);
  parser->GetOutput()->GetChild("Axis")->GetChild("DetectorTranslation")->GetChild("Pz")->GetAllAttributes(listPz,
    true);
  parser->GetOutput()->GetChild("Axis")->GetChild("DetectorRotation")->GetChild("Rx")->GetAllAttributes(listRx, true);
  parser->GetOutput()->GetChild("Axis")->GetChild("DetectorRotation")->GetChild("Ry")->GetAllAttributes(listRy, true);
  parser->GetOutput()->GetChild("Axis")->GetChild("DetectorRotation")->GetChild("Rz")->GetAllAttributes(listRz, true);
  parser->GetOutput()->GetChild("Axis")->GetChild("SourceTranslation")->GetChild("Tx")->GetAllAttributes(listTx, true);
  parser->GetOutput()->GetChild("Axis")->GetChild("SourceTranslation")->GetChild("Ty")->GetAllAttributes(listTy, true);
  parser->GetOutput()->GetChild("Axis")->GetChild("SourceTranslation")->GetChild("Tz")->GetAllAttributes(listTz, true);

  for (unsigned int i = 0; i < 5; i++)
    {
      listPx.pop_front(); Cm.Px[i] = atof(listPx.front().second.c_str());
      listPy.pop_front(); Cm.Py[i] = atof(listPy.front().second.c_str());
      listPz.pop_front(); Cm.Pz[i] = atof(listPz.front().second.c_str());
      listRx.pop_front(); Cm.Rx[i] = atof(listRx.front().second.c_str());
      listRy.pop_front(); Cm.Ry[i] = atof(listRy.front().second.c_str());
      listRz.pop_front(); Cm.Rz[i] = atof(listRz.front().second.c_str());
      listTx.pop_front(); Cm.Tx[i] = atof(listTx.front().second.c_str());
      listTy.pop_front(); Cm.Ty[i] = atof(listTy.front().second.c_str());
      listTz.pop_front(); Cm.Tz[i] = atof(listTz.front().second.c_str());
    }

  Cm.isValid = true;
  return Cm;
}

template< typename TInputImage >
typename ImagXGeometryReader<TInputImage>::CalibrationModelType
ImagXGeometryReader<TInputImage>::GetGeometryForAI1p5FromXMLFiles()
{
  CalibrationModelType C;

  C.isValid = true;

  // Only for AI version less than 2.0
  // Reading iMagX calibration file
  // iMagX axisName parameter
  std::string axisName, sid_s, sdd_s;

  itk::DOMNodeXMLReader::Pointer readerXML = itk::DOMNodeXMLReader::New();
  readerXML->SetFileName(m_CalibrationXMLFileName);
  readerXML->Update();
  itk::DOMNode::Pointer XMLFile = readerXML->GetOutput();

  itk::DOMNode::AttributesListType list;
  itk::DOMNode::AttributesListType::const_iterator list_it;
  itk::DOMNode::ChildrenListType list_child;
  XMLFile->GetAllChildren(list_child);

  for (unsigned int i = 0; i < list_child.size(); i++)
    {
    list_child[i]->GetAllAttributes(list);
    unsigned int k = 0;
    for (list_it = list.begin(); list_it != list.end(); list_it++, k++)
      {
      if ( (*list_it).first.c_str() == std::string("axis") )
        axisName = (*list_it).second.c_str();
      }

    // If cbct axis then extract deformation model parameters
    if (axisName == std::string("CBCT") )
      {
      itk::DOMNode::ChildrenListType list_child2;
      list_child[i]->GetAllChildren(list_child2);
      for (unsigned int l = 0; l < list_child2.size(); l++)
        {
        itk::DOMNode::ChildrenListType list_child3;
        list_child2[l]->GetAllChildren(list_child3);
        unsigned int p = 0;
        for (unsigned int n = 0; n < list_child3.size(); n++, p++)
          {
          itk::DOMNode::AttributesListType list2;
          list_child3[n]->GetAllAttributes(list2);
          unsigned int m = 0;
          for (list_it = list2.begin(); list_it != list2.end(); list_it++, m++)
            {
            if ( (list_child3[n]->GetName() == std::string("Px") ) &&
                 ( (*list_it).first.c_str() != std::string("MSE") ) )
              C.Px[m - 1] = std::atof( (*list_it).second.c_str() );
            else if ( (list_child3[n]->GetName() == std::string("Py") ) &&
                      ( (*list_it).first.c_str() != std::string("MSE") ) )
              C.Py[m - 1] = std::atof( (*list_it).second.c_str() );
            else if ( (list_child3[n]->GetName() == std::string("Pz") ) &&
                      ( (*list_it).first.c_str() != std::string("MSE") ) )
              C.Pz[m - 1] = std::atof( (*list_it).second.c_str() );
            else if ( (list_child3[n]->GetName() == std::string("Rx") ) &&
                      ( (*list_it).first.c_str() != std::string("MSE") ) )
              C.Rx[m - 1] = std::atof( (*list_it).second.c_str() );
            else if ( (list_child3[n]->GetName() == std::string("Ry") ) &&
                      ( (*list_it).first.c_str() != std::string("MSE") ) )
              C.Ry[m - 1] = std::atof( (*list_it).second.c_str() );
            else if ( (list_child3[n]->GetName() == std::string("Rz") ) &&
                      ( (*list_it).first.c_str() != std::string("MSE") ) )
              C.Rz[m - 1] = std::atof( (*list_it).second.c_str() );
            else if ( (list_child3[n]->GetName() == std::string("Tx") ) &&
                      ( (*list_it).first.c_str() != std::string("MSE") ) )
              C.Tx[m - 1] = std::atof( (*list_it).second.c_str() );
            else if ( (list_child3[n]->GetName() == std::string("Ty") ) &&
                      ( (*list_it).first.c_str() != std::string("MSE") ) )
              C.Ty[m - 1] = std::atof( (*list_it).second.c_str() );
            else if ( (list_child3[n]->GetName() == std::string("Tz") ) &&
                      ( (*list_it).first.c_str() != std::string("MSE") ) )
              C.Tz[m - 1] = std::atof( (*list_it).second.c_str() );
            }
          }
        }
      }
    }
  list.clear();
  list_child.clear();

  // Reading iMagX room setup for SID and SDD
  readerXML->SetFileName(m_RoomXMLFileName);
  readerXML->Update();

  XMLFile = readerXML->GetOutput();
  XMLFile->GetAllChildren(list_child);

  for (unsigned int i = 0; i < list_child.size(); i++)
    {
    list_child[i]->GetAllAttributes(list);
    unsigned int k = 0;
    for (list_it = list.begin(); list_it != list.end(); list_it++, k++)
      {
      if ( (*list_it).first.c_str() == std::string("axis") )
        axisName = (*list_it).second.c_str();
      }

    // If cbct axis then extract sdd and sid parameters
    if (axisName == std::string("CBCT") )
      {
      itk::DOMNode::ChildrenListType list_child2;
      list_child[i]->GetAllChildren(list_child2);
      for (unsigned int l = 0; l < list_child2.size(); l++)
        {
        itk::DOMNode::AttributesListType list2;
        list_child2[l]->GetAllAttributes(list2);
        for (list_it = list2.begin(); list_it != list2.end(); list_it++, k++)
          {
          if ( (*list_it).first.c_str() == std::string("sdd") )
            sdd_s = (*list_it).second.c_str();
          if ( (*list_it).first.c_str() == std::string("sid") )
            sid_s = (*list_it).second.c_str();
          }
        }
      }
    }
  C.sid = std::atof(sid_s.c_str() );
  C.sdd = std::atof(sdd_s.c_str() );

  return C;
}

template< typename TInputImage >
std::string ImagXGeometryReader<TInputImage>::getAIversion()
{
  // Create and set ImageIO
  itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(
      m_ProjectionsFileNames[0].c_str(), itk::ImageIOFactory::ReadMode);

  imageIO = itk::GDCMImageIO::New();
  dynamic_cast<itk::GDCMImageIO*>(imageIO.GetPointer() )->LoadPrivateTagsOn();

  // Define, create and set projection reader
  typedef itk::ImageFileReader< TInputImage > ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetImageIO(imageIO);
  reader->SetFileName(m_ProjectionsFileNames[0]);
  reader->UpdateOutputInformation();

  // Read room setup parameters in the DICOM info of the first projection
  std::string AIVersionTagKey = "0018|1020";
  std::string AIVersion = "";
  dynamic_cast<itk::GDCMImageIO*>(imageIO.GetPointer() )->GetValueFromTag(AIVersionTagKey, AIVersion);

  return AIVersion;
}

template< typename TInputImage >
void ImagXGeometryReader<TInputImage>::GenerateData()
{
  const std::string version = getAIversion();

  bool isImagX1p2 = (version.find(m_AI_VERSION_1p2) != std::string::npos);
  bool isImagX1p5 = (version.find(m_AI_VERSION_1p5) != std::string::npos);
  bool isImagX2pX = (version.find(m_AI_VERSION_2pX) != std::string::npos);  
  
  if(!isImagX1p2 && !isImagX1p5 && !isImagX2pX)
    {
    itkExceptionMacro("Unknown AI version : " << version);
    }

  std::string gantryAngleTag;
  FlexmapType flex;
  CalibrationModelType calibModel;
  if (isImagX2pX)
    {
    gantryAngleTag = "300a|011e"; // Warning: CBCT tube angle!
    if (!m_CalibrationXMLFileName.empty() )
      {
      flex = GetGeometryForAI2p1();
      }
    else
      {
      itkExceptionMacro("With AI2.1, you need to provide a calibration file (XML)");
      }
    }
  else if ( (isImagX1p5 || isImagX1p2) && m_CalibrationXMLFileName.empty() && m_RoomXMLFileName.empty() ) // Read geocal from projections
    {
    gantryAngleTag = "300a|011e"; // Nozzle angle
    calibModel = GetGeometryForAI1p5();
    }
  else if ( (isImagX1p5 || isImagX1p2) && !m_CalibrationXMLFileName.empty() && !m_RoomXMLFileName.empty() )
    {
    gantryAngleTag = "300a|011e"; // Nozzle angle
    calibModel = GetGeometryForAI1p5FromXMLFiles();
    }

  // Create new RTK geometry object
  m_Geometry = GeometryType::New();

  // Projection matrices
  for (unsigned int noProj = 0; noProj < m_ProjectionsFileNames.size(); noProj++)
    {
    itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(
        m_ProjectionsFileNames[0].c_str(), itk::ImageIOFactory::ReadMode);
    typedef itk::ImageFileReader< TInputImage > ReaderType;
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(m_ProjectionsFileNames[noProj]);
    imageIO = itk::GDCMImageIO::New();

    reader->SetImageIO(imageIO);
    reader->UpdateOutputInformation();

    // Reading Gantry Angle
    std::string labelId, value;
    itk::GDCMImageIO::GetLabelFromTag(gantryAngleTag, labelId);
    dynamic_cast<itk::GDCMImageIO*>(imageIO.GetPointer() )->GetValueFromTag(gantryAngleTag, value);

    if (isImagX1p5 || isImagX1p2) // Using CalibModel
      {
      float gantryAngle = std::atof(value.c_str() );
      addEntryToGeometry(calibModel, gantryAngle);
      }
    else if (isImagX2pX) // Using flexmap
      {
      float cbctTubeAngle = std::atof(value.c_str() );
      float gantryAngle = cbctTubeAngle + flex.sourceToNozzleOffsetAngle;
      gantryAngle = (gantryAngle > 180.f) ? (gantryAngle - 360.f) : gantryAngle;
      gantryAngle = (gantryAngle < -180.f) ? (gantryAngle + 360.f) : gantryAngle;
      addEntryToGeometry(flex, gantryAngle);
      }
    }
}

template< typename TInputImage >
void
ImagXGeometryReader<TInputImage>::addEntryToGeometry(float gantryAngleDegree,
  float nozzleToRadAngleOffset,
  float sid,
  float sdd,
  std::vector<float>& detTrans,
  std::vector<float>& detRot,
  std::vector<float>& srcTrans)
{
  float deg2rad = float(itk::Math::pi) / 180.f;

  // Conversion from iMagX geometry to RTK geometry standard
  float aid = sid - sdd;
  float sidp = srcTrans[2] + sid;
  float Pzp = detTrans[2] + aid;
  float bx = detRot[0] * deg2rad;
  float by = detRot[1] * deg2rad;
  float bz = detRot[2] * deg2rad;

  float gAngle = gantryAngleDegree + detRot[1];
  float outOfPlaneAngle = detRot[0];
  float inPlaneAngle = detRot[2];
  float sourceToIsocenterDistance = cos(bx)*(cos(by)*sidp + sin(by)*srcTrans[0]) - sin(bx)*srcTrans[1];
  float sourceToDetectorDistance = sourceToIsocenterDistance - cos(bx)*(cos(by)*Pzp + sin(by)*detTrans[0]) + sin(
      bx*detTrans[1]);

  float fac11 = cos(by)*sidp + sin(by)*srcTrans[0];
  float fac12 = sin(bx)*fac11 + cos(bx)*srcTrans[1];
  float fac13 = -sin(by)*sidp + cos(by)*srcTrans[0];
  float sourceOffsetX = sin(bz)*fac12 + cos(bz)*fac13;
  float sourceOffsetY = cos(bz)*fac12 - sin(bz)*fac13;

  float fac21 = cos(by)*Pzp + sin(by)*detTrans[0];
  float fac22 = sin(bx)*fac21 + cos(bx)*detTrans[1];
  float fac23 = -sin(by)*Pzp + cos(by)*detTrans[0];
  float projectionOffsetX = sin(bz)*fac22 + cos(bz)*fac23;
  float projectionOffsetY = cos(bz)*fac22 - sin(bz)*fac23;

  m_Geometry->AddProjection(sourceToIsocenterDistance, sourceToDetectorDistance,
    gAngle + nozzleToRadAngleOffset,
    projectionOffsetX, projectionOffsetY,
    outOfPlaneAngle, inPlaneAngle,
    sourceOffsetX, sourceOffsetY);
}

template< typename TInputImage >
typename ImagXGeometryReader<TInputImage>::InterpResultType
ImagXGeometryReader<TInputImage>::interpolate(const std::vector<float>& flexAngles, bool isCW, float angleDegree)
{
  const int N = static_cast<int>(flexAngles.size() );

  InterpResultType ires;

  ires.id0 = 0;
  ires.id1 = 0;
  ires.a0 = 1.f;
  ires.a1 = 0.f;
  if (N == 1) {
    return ires;
    }

  // Index of the closest angle in flexmap
  int idc = 0;
  float minv = std::numeric_limits<float>::max();
  float delta = 0.f; // to closest angle in flexmap
  for (int i = 0; i < N; ++i) {
    float absdelta = std::abs(angleDegree - flexAngles[i]);
    if (absdelta <= minv) {
      idc = i;
      minv = absdelta;
      delta = angleDegree - flexAngles[i];
      }
    }

  // Clipping
  if (idc == 0 && ( (delta < 0.f && isCW) || (delta > 0.f && !isCW) ) ) {
    ires.id0 = 0;
    ires.id1 = 0;
    ires.a0 = 1.f;
    ires.a1 = 0.f;
    return ires;
    }
  else if (idc == N - 1 && ( (delta > 0.f && isCW) || (delta < 0.f && !isCW) ) ) {
    ires.id0 = N - 1;
    ires.id1 = N - 1;
    ires.a0 = 1.f;
    ires.a1 = 0.f;
    return ires;
    }

  // Interpolation
  if(isCW) {
    if (delta > 0.f) {
      float a = std::abs(delta / (flexAngles[idc + 1] - flexAngles[idc]) );
      ires.id0 = idc;
      ires.id1 = idc + 1;
      ires.a0 = 1.f - a;
      ires.a1 = a;
      }
    else if (delta < 0.f) {
      float a = std::abs(delta / (flexAngles[idc] - flexAngles[idc - 1]) );
      ires.id0 = idc - 1;
      ires.id1 = idc;
      ires.a0 = a;
      ires.a1 = 1.f - a;
      }
    }
  else {
    if (delta < 0.f) {
      float a = std::abs(delta / (flexAngles[idc + 1] - flexAngles[idc]) );
      ires.id0 = idc;
      ires.id1 = idc + 1;
      ires.a0 = 1.f - a;
      ires.a1 = a;
      }
    else if (delta > 0.f) {
      float a = std::abs(delta / (flexAngles[idc] - flexAngles[idc - 1]) );
      ires.id0 = idc - 1;
      ires.id1 = idc;
      ires.a0 = a;
      ires.a1 = 1.f - a;
      }
    }

  return ires;
}

template< typename TInputImage >
std::vector<float>
ImagXGeometryReader<TInputImage>::getInterpolatedValue(const InterpResultType& ires,
  const std::vector<float>& Dx,
  const std::vector<float>& Dy,
  const std::vector<float>& Dz)
{
  std::vector<float> d;

  d.resize(3);
  d[0] = ires.a0 * Dx[ires.id0] + ires.a1 * Dx[ires.id1];
  d[1] = ires.a0 * Dy[ires.id0] + ires.a1 * Dy[ires.id1];
  d[2] = ires.a0 * Dz[ires.id0] + ires.a1 * Dz[ires.id1];
  return d;
}

template< typename TInputImage >
void
ImagXGeometryReader<TInputImage>::addEntryToGeometry(const FlexmapType& f, float gantryAngle)
{
  // Deformation obtained by sampling the flexmap

  InterpResultType ires = interpolate(f.anglesDeg, f.isCW, gantryAngle);

  // Detector translation deformations
  std::vector<float> detTrans = this->getInterpolatedValue(ires, f.Px, f.Py, f.Pz);

  // Detector rotation deformations
  std::vector<float> detRot = this->getInterpolatedValue(ires, f.Rx, f.Ry, f.Rz);

  // Source translation deformations
  std::vector<float> srcTrans = this->getInterpolatedValue(ires, f.Tx, f.Ty, f.Tz);

  // Add new entry to RTK geometry
  addEntryToGeometry(gantryAngle, f.sourceToNozzleOffsetAngle, f.sid, f.sdd, detTrans, detRot, srcTrans);
}

template< typename TInputImage >
std::vector<float>
ImagXGeometryReader<TInputImage>::getDeformations(float gantryAngle,
  const std::vector<float>& Dx,
  const std::vector<float>& Dy,
  const std::vector<float>& Dz)
{
  std::vector<float> d;

  d.resize(3);
  float gRad = gantryAngle*std::acos(-1.f) / 180.f;
  d[0] = Dx[0] + Dx[1] * gantryAngle + Dx[2] * std::cos(Dx[3] * gRad + Dx[4]);
  d[1] = Dy[0] + Dy[1] * gantryAngle + Dy[2] * std::cos(Dy[3] * gRad + Dy[4]);
  d[2] = Dz[0] + Dz[1] * gantryAngle + Dz[2] * std::cos(Dz[3] * gRad + Dz[4]);
  return d;
}

template< typename TInputImage >
void
ImagXGeometryReader<TInputImage>::addEntryToGeometry(const CalibrationModelType& c, float gantryAngle)
{
  // Deformation computation following model: (a0 + a1*t) + a2*cos(a3*t + a4)

  // Detector translation deformations
  std::vector<float> detTrans = this->getDeformations(gantryAngle, c.Px, c.Py, c.Pz);

  // Detector rotation deformations
  std::vector<float> detRot = this->getDeformations(gantryAngle, c.Rx, c.Ry, c.Rz);

  // Source translation deformations
  std::vector<float> srcTrans = this->getDeformations(gantryAngle, c.Tx, c.Ty, c.Tz);

  // Add new entry to RTK geometry
  addEntryToGeometry(gantryAngle, c.sourceToNozzleOffsetAngle, c.sid, c.sdd, detTrans, detRot, srcTrans);
}
} //namespace rtk
#endif
