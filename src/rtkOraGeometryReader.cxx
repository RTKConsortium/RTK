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

#ifndef rtkOraGeometryReader_hxx
#define rtkOraGeometryReader_hxx

#include "rtkMacro.h"
#include "rtkOraGeometryReader.h"
#include "rtkOraXMLFileReader.h"
#include "rtkIOFactories.h"

#include <itkImageIOBase.h>
#include <itkImageIOFactory.h>

namespace rtk
{


void
OraGeometryReader::GenerateData()
{
  m_Geometry = GeometryType::New();
  RegisterIOFactories();
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

    using MetaDataVectorType = itk::MetaDataObject<VectorType>;
    using MetaDataMatrixType = itk::MetaDataObject<Matrix3x3Type>;
    using MetaDataDoubleType = itk::MetaDataObject<double>;

    // Source position
    MetaDataVectorType * spMeta = dynamic_cast<MetaDataVectorType *>(dic["SourcePosition"].GetPointer());
    if (spMeta == nullptr)
    {
      itkExceptionMacro(<< "No SourcePosition in " << projectionsFileName);
    }
    PointType sp(&(spMeta->GetMetaDataObjectValue()[0]));

    // Origin (detector position)
    MetaDataVectorType * dpMeta = dynamic_cast<MetaDataVectorType *>(dic["Origin"].GetPointer());
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

    // Got it, add to geometry
    m_Geometry->AddProjection(sp, dp, VectorType(&(mat[0][0])), VectorType(&(mat[1][0])));

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
