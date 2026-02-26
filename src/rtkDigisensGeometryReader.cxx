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

#include "rtkDigisensGeometryReader.h"
#include "rtkDigisensGeometryXMLFileReader.h"

#include <itkMacro.h>
#include <itkMetaDataObject.h>
#include <itkVersor.h>
#include <itkCenteredEuler3DTransform.h>

rtk::DigisensGeometryReader ::DigisensGeometryReader() = default;

void
rtk::DigisensGeometryReader ::GenerateData()
{
  // Create new RTK geometry object
  m_Geometry = GeometryType::New();

  // Read Varian XML file (for common geometric information)
  rtk::DigisensGeometryXMLFileReader::Pointer digisensXmlReader;
  digisensXmlReader = rtk::DigisensGeometryXMLFileReader::New();
  digisensXmlReader->SetFilename(m_XMLFileName);
  digisensXmlReader->GenerateOutputInformation();

  // Constants used to generate projection matrices
  itk::MetaDataDictionary & dic = *(digisensXmlReader->GetOutputObject());

  // Getting elements positions
  using MetaDataVectorType = itk::MetaDataObject<GeometryType::VectorType>;
  auto * rotationAxisMetaData = dynamic_cast<MetaDataVectorType *>(dic["ROTATIONaxis"].GetPointer());
  if (rotationAxisMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"ROTATIONaxis\".");
  GeometryType::VectorType rotationAxis = rotationAxisMetaData->GetMetaDataObjectValue();

  auto * rotationCenterMetaData = dynamic_cast<MetaDataVectorType *>(dic["ROTATIONcenter"].GetPointer());
  if (rotationCenterMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"ROTATIONcenter\".");
  GeometryType::VectorType rotationCenter = rotationCenterMetaData->GetMetaDataObjectValue();

  auto * sourcePositionMetaData = dynamic_cast<MetaDataVectorType *>(dic["XRAYsource"].GetPointer());
  if (sourcePositionMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"XRAYsource\".");
  GeometryType::VectorType sourcePosition = sourcePositionMetaData->GetMetaDataObjectValue();

  auto * detectorPositionMetaData = dynamic_cast<MetaDataVectorType *>(dic["CAMERAreference"].GetPointer());
  if (detectorPositionMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"CAMERAreference\".");
  GeometryType::VectorType detectorPosition = detectorPositionMetaData->GetMetaDataObjectValue();

  auto * detectorNormalMetaData = dynamic_cast<MetaDataVectorType *>(dic["CAMERAnormal"].GetPointer());
  if (detectorNormalMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"CAMERAnormal\".");
  GeometryType::VectorType detectorNormal = detectorNormalMetaData->GetMetaDataObjectValue();

  auto * detectorHorizontalMetaData = dynamic_cast<MetaDataVectorType *>(dic["CAMERAhorizontal"].GetPointer());
  if (detectorHorizontalMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"CAMERAhorizontal\".");
  GeometryType::VectorType detectorHorizontal = detectorHorizontalMetaData->GetMetaDataObjectValue();

  auto * detectorVerticalMetaData = dynamic_cast<MetaDataVectorType *>(dic["CAMERAvertical"].GetPointer());
  if (detectorVerticalMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"CAMERAvertical\".");
  GeometryType::VectorType detectorVertical = detectorVerticalMetaData->GetMetaDataObjectValue();

  // Check assumptions
  if (sourcePosition[0] != 0. || sourcePosition[1] != 0. || detectorNormal[0] != 0. || detectorNormal[1] != 0. ||
      detectorHorizontal[1] != 0. || detectorHorizontal[2] != 0. || detectorVertical[0] != 0. ||
      detectorVertical[2] != 0.)
  {
    itkGenericExceptionMacro(<< "Geometric assumptions not verified");
  }

  // Source / Detector / Center distances
  double sdd = itk::Math::Absolute(sourcePosition[2] - detectorPosition[2]);
  double sid = itk::Math::Absolute(sourcePosition[2] - rotationCenter[2]);

  // Scaling
  using MetaDataIntegerType = itk::MetaDataObject<int>;
  // int pixelWidth = dynamic_cast<MetaDataIntegerType *>(dic["CAMERApixelWidth"].GetPointer()
  // )->GetMetaDataObjectValue(); int pixelHeight = dynamic_cast<MetaDataIntegerType
  // *>(dic["CAMERApixelHeight"].GetPointer() )->GetMetaDataObjectValue();
  using MetaDataDoubleType = itk::MetaDataObject<double>;
  // double totalWidth = dynamic_cast<MetaDataDoubleType *>(dic["CAMERAtotalWidth"].GetPointer()
  // )->GetMetaDataObjectValue(); double totalHeight = dynamic_cast<MetaDataDoubleType
  // *>(dic["CAMERAtotalHeight"].GetPointer() )->GetMetaDataObjectValue(); double projectionScalingX =
  // detectorHorizontal[0] * totalWidth / (pixelWidth-1); double projectionScalingY = detectorVertical[1] * totalHeight
  // / (pixelHeight-1);

  // Projection offset: the offset is given in the volume coordinate system =>
  // convert to
  double projectionOffsetX = detectorPosition[0] * detectorHorizontal[0];
  double projectionOffsetY = detectorPosition[1] * detectorVertical[1];

  // Rotation
  auto * startAngleMetaData = dynamic_cast<MetaDataDoubleType *>(dic["RADIOSstartAngle"].GetPointer());
  if (startAngleMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"RADIOSstartAngle\".");
  const double startAngle = startAngleMetaData->GetMetaDataObjectValue();

  auto * angularRangeMetaData = dynamic_cast<MetaDataDoubleType *>(dic["RADIOSangularRange"].GetPointer());
  if (angularRangeMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"RADIOSangularRange\".");
  const double angularRange = angularRangeMetaData->GetMetaDataObjectValue();

  auto * nProjMetaData = dynamic_cast<MetaDataIntegerType *>(dic["RADIOSNumberOfFiles"].GetPointer());
  if (nProjMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"RADIOSNumberOfFiles\".");
  const int nProj = nProjMetaData->GetMetaDataObjectValue();
  for (int i = 0; i < nProj; i++)
  {
    // Convert rotation center and rotation axis parameterization to euler angles
    double angle = -startAngle - i * angularRange / nProj;

    const double        degreesToRadians = std::atan(1.0) / 45.0;
    itk::Versor<double> xfm3DVersor;
    xfm3DVersor.Set(rotationAxis, angle * degreesToRadians);

    auto xfm3D = itk::CenteredEuler3DTransform<double>::New();
    xfm3D->SetMatrix(xfm3DVersor.GetMatrix());

    m_Geometry->AddProjectionInRadians(sid,
                                       sdd,
                                       xfm3D->GetAngleY(),
                                       projectionOffsetX,
                                       projectionOffsetY,
                                       xfm3D->GetAngleX(),
                                       xfm3D->GetAngleZ(),
                                       rotationCenter[0],
                                       rotationCenter[1]);
  }
}
