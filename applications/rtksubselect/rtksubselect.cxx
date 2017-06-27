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

#include "rtksubselect_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkProjectionsReader.h"
#include "rtkConstantImageSource.h"

#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkPasteImageFilter.h>

int main(int argc, char * argv[])
{
  GGO(rtksubselect, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtksubselect>(reader, args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )

  // Geometry
  if(args_info.verbose_flag)
    std::cout << "Reading geometry information from "
              << args_info.geometry_arg
              << "..."
              << std::endl;
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() )

  // Compute the indices of the selected projections
  std::vector<int> indices;
  int n = geometryReader->GetOutputObject()->GetGantryAngles().size();
  if(args_info.last_given)
    n = std::min(args_info.last_arg, n);
  if(args_info.list_given)
    for(unsigned int i=0; i<args_info.list_given; i++)
      {
        indices.push_back(args_info.list_arg[i]);
      }
  else
    for(int noProj=args_info.first_arg; noProj<n; noProj+=args_info.step_arg)
      {
        indices.push_back(noProj);
      }

  // Output RTK geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer outputGeometry = GeometryType::New();

  // Output projections object
  typedef rtk::ConstantImageSource< OutputImageType > SourceType;
  SourceType::Pointer source = SourceType::New();
  source->SetInformationFromImage(reader->GetOutput());
  OutputImageType::SizeType outputSize = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
  outputSize[Dimension - 1] = indices.size();
  source->SetSize(outputSize);
  source->SetConstant(0);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( source->Update() )

  // Fill in the outputGeometry and the output projections
  typedef itk::PasteImageFilter<OutputImageType> PasteType;
  PasteType::Pointer paste = PasteType::New();
  paste->SetSourceImage(reader->GetOutput());
  paste->SetDestinationImage(source->GetOutput());

  OutputImageType::RegionType sourceRegion;
  OutputImageType::IndexType destinationIndex;
  for (unsigned int i=0; i<indices.size(); i++)
    {
    // If it is not the first projection, we need to use the output of
    // the paste filter as input
    if(i)
      {
      OutputImageType::Pointer pimg = paste->GetOutput();
      pimg->DisconnectPipeline();
      paste->SetDestinationImage(pimg);
      }

    sourceRegion = reader->GetOutput()->GetLargestPossibleRegion();
    sourceRegion.SetIndex(Dimension - 1, indices[i]);
    sourceRegion.SetSize(Dimension - 1, 1);
    paste->SetSourceRegion(sourceRegion);

    destinationIndex = reader->GetOutput()->GetLargestPossibleRegion().GetIndex();
    destinationIndex[Dimension -1] = i;
    paste->SetDestinationIndex(destinationIndex);

    TRY_AND_EXIT_ON_ITK_EXCEPTION( paste->Update() )

    // Fill in the output geometry object
    outputGeometry->SetRadiusCylindricalDetector(geometryReader->GetOutputObject()->GetRadiusCylindricalDetector());
    outputGeometry->AddProjectionInRadians(geometryReader->GetOutputObject()->GetSourceToIsocenterDistances()[indices[i]],
                                           geometryReader->GetOutputObject()->GetSourceToDetectorDistances()[indices[i]],
                                           geometryReader->GetOutputObject()->GetGantryAngles()[indices[i]],
                                           geometryReader->GetOutputObject()->GetProjectionOffsetsX()[indices[i]],
                                           geometryReader->GetOutputObject()->GetProjectionOffsetsY()[indices[i]],
                                           geometryReader->GetOutputObject()->GetOutOfPlaneAngles()[indices[i]],
                                           geometryReader->GetOutputObject()->GetInPlaneAngles()[indices[i]],
                                           geometryReader->GetOutputObject()->GetSourceOffsetsX()[indices[i]],
                                           geometryReader->GetOutputObject()->GetSourceOffsetsY()[indices[i]]);
    outputGeometry->SetCollimationOfLastProjection(geometryReader->GetOutputObject()->GetCollimationUInf()[indices[i]],
                                                   geometryReader->GetOutputObject()->GetCollimationUSup()[indices[i]],
                                                   geometryReader->GetOutputObject()->GetCollimationVInf()[indices[i]],
                                                   geometryReader->GetOutputObject()->GetCollimationVSup()[indices[i]] );
    }

  // Geometry writer
  if(args_info.verbose_flag)
    std::cout << "Writing geometry information in "
              << args_info.out_geometry_arg
              << "..."
              << std::endl;
  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter =
    rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.out_geometry_arg);
  xmlWriter->SetObject( &(*outputGeometry) );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( xmlWriter->WriteFile() )

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.out_proj_arg );
  writer->SetInput( paste->GetOutput() );
  //TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->UpdateOutputInformation() )
//  writer->SetNumberOfStreamDivisions( 1 + reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() / (1024*1024*4) );

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
