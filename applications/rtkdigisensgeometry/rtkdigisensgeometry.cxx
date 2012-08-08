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

#include "rtkdigisensgeometry_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkDigisensGeometryXMLFileReader.h"

#include <itkMetaDataObject.h>
#include <itkCenteredEuler3DTransform.h>

#include <iomanip>

int main(int argc, char * argv[])
{
  GGO(rtkdigisensgeometry, args_info);

  // RTK geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();

  // Read Varian XML file (for common geometric information)
  rtk::DigisensGeometryXMLFileReader::Pointer digisensXmlReader;
  digisensXmlReader = rtk::DigisensGeometryXMLFileReader::New();
  digisensXmlReader->SetFilename(args_info.xml_file_arg);
  digisensXmlReader->GenerateOutputInformation();

  // Constants used to generate projection matrices
  itk::MetaDataDictionary &dic = *(digisensXmlReader->GetOutputObject() );

  // Getting elements positions
  typedef itk::MetaDataObject< GeometryType::VectorType > MetaDataVectorType;
  GeometryType::VectorType rotationAxis       =
    dynamic_cast<MetaDataVectorType *>(dic["ROTATIONaxis"].GetPointer() )->GetMetaDataObjectValue();
  GeometryType::VectorType rotationCenter     =
    dynamic_cast<MetaDataVectorType *>(dic["ROTATIONcenter"].GetPointer() )->GetMetaDataObjectValue();
  GeometryType::VectorType sourcePosition     =
    dynamic_cast<MetaDataVectorType *>(dic["XRAYsource"].GetPointer() )->GetMetaDataObjectValue();
  GeometryType::VectorType detectorPosition   =
    dynamic_cast<MetaDataVectorType *>(dic["CAMERAreference"].GetPointer() )->GetMetaDataObjectValue();
  GeometryType::VectorType detectorNormal     =
    dynamic_cast<MetaDataVectorType *>(dic["CAMERAnormal"].GetPointer() )->GetMetaDataObjectValue();
  GeometryType::VectorType detectorHorizontal =
    dynamic_cast<MetaDataVectorType *>(dic["CAMERAhorizontal"].GetPointer() )->GetMetaDataObjectValue();
  GeometryType::VectorType detectorVertical   =
    dynamic_cast<MetaDataVectorType *>(dic["CAMERAvertical"].GetPointer() )->GetMetaDataObjectValue();

  // Check assumptions
  if(sourcePosition[0] != 0. ||
     sourcePosition[1] != 0. ||
     detectorNormal[0] != 0. ||
     detectorNormal[1] != 0. ||
     detectorHorizontal[1] != 0. ||
     detectorHorizontal[2] != 0. ||
     detectorVertical[0] != 0. ||
     detectorVertical[2] != 0.) {
    std::cerr << "Geometric assumptions not verified" << std::endl;
    return EXIT_FAILURE;
    }

  // Source / Detector / Center distances
  double sdd = fabs(sourcePosition[2] - detectorPosition[2]);
  double sid = fabs(sourcePosition[2] - rotationCenter[2]);

  // Scaling
  typedef itk::MetaDataObject< int > MetaDataIntegerType;
  int pixelWidth = dynamic_cast<MetaDataIntegerType *>(dic["CAMERApixelWidth"].GetPointer() )->GetMetaDataObjectValue();
  int pixelHeight = dynamic_cast<MetaDataIntegerType *>(dic["CAMERApixelHeight"].GetPointer() )->GetMetaDataObjectValue();
  typedef itk::MetaDataObject< double > MetaDataDoubleType;
  double totalWidth = dynamic_cast<MetaDataDoubleType *>(dic["CAMERAtotalWidth"].GetPointer() )->GetMetaDataObjectValue();
  double totalHeight =
    dynamic_cast<MetaDataDoubleType *>(dic["CAMERAtotalHeight"].GetPointer() )->GetMetaDataObjectValue();
  double projectionScalingX = detectorHorizontal[0] * totalWidth / (pixelWidth-1);
  double projectionScalingY = detectorVertical[1] * totalHeight / (pixelHeight-1);
  DD(projectionScalingX)
  DD(projectionScalingY)

  // Projection offset: the offset is given in the volume coordinate system =>
  // convert to
  double projectionOffsetX = detectorPosition[0] * detectorHorizontal[0];
  double projectionOffsetY = detectorPosition[1] * detectorVertical[1];

  // Rotation
  double startAngle =
    dynamic_cast<MetaDataDoubleType *>(dic["RADIOSstartAngle"].GetPointer() )->GetMetaDataObjectValue();
  double angularRange =
    dynamic_cast<MetaDataDoubleType *>(dic["RADIOSangularRange"].GetPointer() )->GetMetaDataObjectValue();
  int nProj = dynamic_cast<MetaDataIntegerType *>(dic["RADIOSNumberOfFiles"].GetPointer() )->GetMetaDataObjectValue();
  for(int i=0; i<nProj; i++ )
    {
    // Convert rotation center and rotation axis parameterization to euler angles
    double angle = - startAngle - i * angularRange / nProj;

    const double degreesToRadians = vcl_atan(1.0) / 45.0;
    itk::Versor<double> xfm3DVersor;
    xfm3DVersor.Set(rotationAxis, angle*degreesToRadians);

    typedef itk::CenteredEuler3DTransform<double> ThreeDTransformType;
    ThreeDTransformType::Pointer xfm3D = ThreeDTransformType::New();
    xfm3D->SetMatrix(xfm3DVersor.GetMatrix());

    geometry->AddProjection( sid,
                             sdd,
                             xfm3D->GetAngleY(),
                             projectionOffsetX,
                             projectionOffsetY,
                             xfm3D->GetAngleX(),
                             xfm3D->GetAngleZ(),
                             rotationCenter[0],
                             rotationCenter[1]);
    }

  // Write
  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter =
    rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject(&(*geometry) );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( xmlWriter->WriteFile() )

  return EXIT_SUCCESS;
}
