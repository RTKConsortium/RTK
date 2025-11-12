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

#ifndef rtkOraGeometryReader_hxx
#define rtkOraGeometryReader_hxx

#include "rtkMacro.h"
#include "rtkOraGeometryReader.h"
#include "rtkOraXMLFileReader.h"
#include "rtkIOFactories.h"

#include <itkImageIOBase.h>
#include <itkImageIOFactory.h>
#include <itkVersorRigid3DTransform.h>
#include <itkQuaternionRigidTransform.h>

namespace rtk
{


void
OraGeometryReader::GenerateData()
{
  m_Geometry = GeometryType::New();
  RegisterIOFactories();
  itk::QuaternionRigidTransform<double>::Pointer firstQuaternionsX{ nullptr };
  itk::Vector<double, 3>                         firstTranslation{ 0.0 };
  for (const std::string & projectionsFileName : m_ProjectionsFileNames)
  {
    itk::ImageIOBase::Pointer reader;

    reader =
      itk::ImageIOFactory::CreateImageIO(projectionsFileName.c_str(), itk::ImageIOFactory::IOFileModeEnum::ReadMode);
    if (!reader)
    {
      itkExceptionMacro("Error reading file " << projectionsFileName);
    }
    reader->SetFileName(projectionsFileName.c_str());
    reader->ReadImageInformation();
    itk::MetaDataDictionary & dic = reader->GetMetaDataDictionary();

    using MetaDataPointType = itk::MetaDataObject<PointType>;
    using MetaDataMatrixType = itk::MetaDataObject<Matrix3x3Type>;
    using MetaDataDoubleType = itk::MetaDataObject<double>;
    using MetaDataVectorDoubleType = itk::MetaDataObject<std::vector<double>>;
    using MetaDataVectorIntType = itk::MetaDataObject<std::vector<int>>;

    // Source position
    MetaDataPointType * spMeta = dynamic_cast<MetaDataPointType *>(dic["SourcePosition"].GetPointer());
    if (spMeta == nullptr)
    {
      itkExceptionMacro(<< "No SourcePosition in " << projectionsFileName);
    }
    PointType sp(&(spMeta->GetMetaDataObjectValue()[0]));

    // Origin (detector position)
    MetaDataPointType * dpMeta = dynamic_cast<MetaDataPointType *>(dic["Origin"].GetPointer());
    if (dpMeta == nullptr)
    {
      itkExceptionMacro(<< "No Origin in " << projectionsFileName);
    }
    PointType dp(&(dpMeta->GetMetaDataObjectValue()[0]));

    // Direction (detector orientation)
    MetaDataMatrixType * matMeta = dynamic_cast<MetaDataMatrixType *>(dic["Direction"].GetPointer());
    if (matMeta == nullptr)
    {
      itkExceptionMacro(<< "No Direction in " << projectionsFileName);
    }
    Matrix3x3Type mat = matMeta->GetMetaDataObjectValue();
    VectorType    u = VectorType(&(mat[0][0]));
    VectorType    v = VectorType(&(mat[1][0]));

    // table_axis_distance_cm
    MetaDataDoubleType * thMeta = dynamic_cast<MetaDataDoubleType *>(dic["table_axis_distance_cm"].GetPointer());
    if (thMeta == nullptr)
    {
      itkExceptionMacro(<< "No table_axis_distance_cm in " << projectionsFileName);
    }
    double th = thMeta->GetMetaDataObjectValue();
    sp[2] -= th * 10.;
    dp[2] -= th * 10.;

    // longitudinalposition_cm
    MetaDataDoubleType * axMeta = dynamic_cast<MetaDataDoubleType *>(dic["longitudinalposition_cm"].GetPointer());
    if (axMeta == nullptr)
    {
      itkExceptionMacro(<< "No longitudinalposition_cm in " << projectionsFileName);
    }
    double ax = axMeta->GetMetaDataObjectValue();
    sp[1] -= ax * 10.;
    dp[1] -= ax * 10.;

    // Ring tilt (only available in some versions)
    MetaDataDoubleType * tiltLeftMeta = dynamic_cast<MetaDataDoubleType *>(dic["tiltleft_deg"].GetPointer());
    if (tiltLeftMeta != nullptr && m_OptiTrackObjectID < 0)
    {
      double               tiltLeft = tiltLeftMeta->GetMetaDataObjectValue();
      MetaDataDoubleType * tiltRightMeta = dynamic_cast<MetaDataDoubleType *>(dic["tiltright_deg"].GetPointer());
      double               tiltRight = tiltRightMeta->GetMetaDataObjectValue();
      auto                 tiltTransform = itk::VersorRigid3DTransform<double>::New();
      const double         deg2rad = std::atan(1.0) / 45.0;
      tiltTransform->SetRotation(itk::MakeVector(1., 0., 0.), 0.5 * (tiltLeft + tiltRight) * deg2rad);

      // Set center of rotation
      MetaDataDoubleType * yvecMeta =
        dynamic_cast<MetaDataDoubleType *>(dic["ydistancebaseunitcs2imagingcs_cm"].GetPointer());
      double               yvec = yvecMeta->GetMetaDataObjectValue();
      MetaDataDoubleType * zvecMeta =
        dynamic_cast<MetaDataDoubleType *>(dic["zdistancebaseunitcs2imagingcs_cm"].GetPointer());
      double zvec = zvecMeta->GetMetaDataObjectValue();
      tiltTransform->SetCenter(itk::MakePoint(0., -10. * yvec, -10. * zvec));

      sp = tiltTransform->TransformPoint(sp);
      dp = tiltTransform->TransformPoint(dp);
      u = tiltTransform->TransformVector(u);
      v = tiltTransform->TransformVector(v);
    }

    // Ring yaw (only available in some versions)
    MetaDataDoubleType * yawMeta = dynamic_cast<MetaDataDoubleType *>(dic["room_cs_yaw_deg"].GetPointer());
    if (yawMeta != nullptr && m_OptiTrackObjectID < 0)
    {
      double       yaw = yawMeta->GetMetaDataObjectValue();
      auto         tiltTransform = itk::VersorRigid3DTransform<double>::New();
      const double deg2rad = std::atan(1.0) / 45.0;
      tiltTransform->SetRotation(itk::MakeVector(0., 0., 1.), yaw * deg2rad);

      // Set center of rotation
      MetaDataDoubleType * yvecMeta =
        dynamic_cast<MetaDataDoubleType *>(dic["ydistancebaseunitcs2imagingcs_cm"].GetPointer());
      double yvec = yvecMeta->GetMetaDataObjectValue();
      tiltTransform->SetCenter(itk::MakePoint(0., -10. * yvec, 0.));

      sp = tiltTransform->TransformPoint(sp);
      dp = tiltTransform->TransformPoint(dp);
      u = tiltTransform->TransformVector(u);
      v = tiltTransform->TransformVector(v);
    }

    // OptiTrack objects (objects tracked with infrared cameras)
    if (m_OptiTrackObjectID >= 0)
    {
      // Find ID index of the OptiTrack object
      MetaDataVectorIntType * idsMeta = dynamic_cast<MetaDataVectorIntType *>(dic["optitrack_object_ids"].GetPointer());
      if (idsMeta == nullptr)
        itkExceptionMacro("Could not find optitrack_object_ids in " << projectionsFileName);
      const std::vector<int> ids = idsMeta->GetMetaDataObjectValue();
      auto                   idIt = std::find(ids.begin(), ids.end(), m_OptiTrackObjectID);
      unsigned int           idIdx = idIt - ids.begin();

      // Translation
      MetaDataVectorDoubleType * posMeta =
        dynamic_cast<MetaDataVectorDoubleType *>(dic["optitrack_positions"].GetPointer());
      if (posMeta == nullptr)
        itkExceptionMacro("Could not find optitrack_positions in " << projectionsFileName);
      const std::vector<double> p = posMeta->GetMetaDataObjectValue();
      if (p.size() < 3 * (idIdx + 1))
        itkExceptionMacro("Not enough values in optitrack_positions of " << projectionsFileName);
      itk::Vector<double, 3> translation = 10. * itk::MakeVector(p[idIdx * 3], p[idIdx * 3 + 1], p[idIdx * 3 + 2]);

      // Rotation
      MetaDataVectorDoubleType * rotMeta =
        dynamic_cast<MetaDataVectorDoubleType *>(dic["optitrack_rotations"].GetPointer());
      if (rotMeta == nullptr)
        itkExceptionMacro("Could not find optitrack_rotations in " << projectionsFileName);
      const std::vector<double> optitrackRotations = rotMeta->GetMetaDataObjectValue();
      if (optitrackRotations.size() < 4 * (idIdx + 1))
        itkExceptionMacro("Not enough values in optitrack_rotations of " << projectionsFileName);
      auto                                                  quaternionsX = itk::QuaternionRigidTransform<double>::New();
      itk::QuaternionRigidTransform<double>::ParametersType quaternionsXParam(7);
      quaternionsXParam[3] = optitrackRotations[idIdx * 4];
      quaternionsXParam[0] = optitrackRotations[idIdx * 4 + 1];
      quaternionsXParam[1] = optitrackRotations[idIdx * 4 + 2];
      quaternionsXParam[2] = optitrackRotations[idIdx * 4 + 3];
      quaternionsXParam[4] = 0.;
      quaternionsXParam[5] = 0.;
      quaternionsXParam[6] = 0.;
      quaternionsX->SetParameters(quaternionsXParam);

      if (firstQuaternionsX.GetPointer() == nullptr)
      {
        firstQuaternionsX = quaternionsX;
        firstTranslation = translation;
      }
      else
      {
        itk::MatrixOffsetTransformBase<double, 3, 3>::InverseTransformBasePointer invQuaternionsX =
          quaternionsX->GetInverseTransform();

        sp = sp - translation;
        dp = dp - translation;
        sp = invQuaternionsX->TransformPoint(sp);
        dp = invQuaternionsX->TransformPoint(dp);
        u = invQuaternionsX->TransformVector(u);
        v = invQuaternionsX->TransformVector(v);

        sp = firstQuaternionsX->TransformPoint(sp);
        dp = firstQuaternionsX->TransformPoint(dp);
        u = firstQuaternionsX->TransformVector(u);
        v = firstQuaternionsX->TransformVector(v);
        sp = sp + firstTranslation;
        dp = dp + firstTranslation;
      }
    }

    // Got it, add to geometry
    if (!m_Geometry->AddProjection(sp, dp, u, v))
    {
      itkWarningMacro("Could not add " << projectionsFileName << " with sp=" << sp << ", dp=" << dp << ", u=" << u
                                       << " and v=" << v);
    }

    // Now add the collimation
    // longitudinalposition_cm
    double               uinf = std::numeric_limits<double>::max();
    MetaDataDoubleType * uinfMeta = dynamic_cast<MetaDataDoubleType *>(dic["xrayx1_cm"].GetPointer());
    if (uinfMeta != nullptr)
    {
      uinf = 10. * uinfMeta->GetMetaDataObjectValue() + m_CollimationMargin[0];
    }

    double               usup = std::numeric_limits<double>::max();
    MetaDataDoubleType * usupMeta = dynamic_cast<MetaDataDoubleType *>(dic["xrayx2_cm"].GetPointer());
    if (usupMeta != nullptr)
    {
      usup = 10. * usupMeta->GetMetaDataObjectValue() + m_CollimationMargin[1];
    }

    double               vinf = std::numeric_limits<double>::max();
    MetaDataDoubleType * vinfMeta = dynamic_cast<MetaDataDoubleType *>(dic["xrayy1_cm"].GetPointer());
    if (vinfMeta != nullptr)
    {
      vinf = 10. * vinfMeta->GetMetaDataObjectValue() + m_CollimationMargin[2];
    }

    double               vsup = std::numeric_limits<double>::max();
    MetaDataDoubleType * vsupMeta = dynamic_cast<MetaDataDoubleType *>(dic["xrayy2_cm"].GetPointer());
    if (vsupMeta != nullptr)
    {
      vsup = 10. * vsupMeta->GetMetaDataObjectValue() + m_CollimationMargin[3];
    }
    m_Geometry->SetCollimationOfLastProjection(uinf, usup, vinf, vsup);
  }
}
} // namespace rtk
#endif
