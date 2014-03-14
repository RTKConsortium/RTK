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

#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>

int main(int argc, char * argv[])
{
  GGO(rtksubselect, args_info);

  // Generate file names
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(args_info.nsort_flag);
  names->SetRegularExpression(args_info.regexp_arg);
  names->SetSubMatch(args_info.submatch_arg);

  if(args_info.verbose_flag)
    std::cout << "Regular expression matches "
              << names->GetFileNames().size()
              << " file(s)..."
              << std::endl;

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

  // Output RTK geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer outputGeometry = GeometryType::New();

  // Output filenames
  std::vector< std::string > outputProjectionsNames;

  // Subsection
  int n = geometryReader->GetOutputObject()->GetGantryAngles().size();
  if(args_info.list_given)
    {
    for(unsigned int i=0; i<args_info.list_given; i++)
      {
      int noProj = args_info.list_arg[i];
      outputGeometry->AddProjection(geometryReader->GetOutputObject()->GetSourceToIsocenterDistances()[noProj],
                                    geometryReader->GetOutputObject()->GetSourceToDetectorDistances()[noProj],
                                    geometryReader->GetOutputObject()->GetGantryAngles()[noProj],
                                    geometryReader->GetOutputObject()->GetProjectionOffsetsX()[noProj],
                                    geometryReader->GetOutputObject()->GetProjectionOffsetsY()[noProj],
                                    geometryReader->GetOutputObject()->GetOutOfPlaneAngles()[noProj],
                                    geometryReader->GetOutputObject()->GetInPlaneAngles()[noProj],
                                    geometryReader->GetOutputObject()->GetSourceOffsetsX()[noProj],
                                    geometryReader->GetOutputObject()->GetSourceOffsetsY()[noProj]);
      outputProjectionsNames.push_back(names->GetFileNames()[noProj]);
      }
    }
  else
    for(int noProj=args_info.first_arg; noProj<n; noProj+=args_info.step_arg)
      {
      outputGeometry->AddProjection(geometryReader->GetOutputObject()->GetSourceToIsocenterDistances()[noProj],
                                    geometryReader->GetOutputObject()->GetSourceToDetectorDistances()[noProj],
                                    geometryReader->GetOutputObject()->GetGantryAngles()[noProj],
                                    geometryReader->GetOutputObject()->GetProjectionOffsetsX()[noProj],
                                    geometryReader->GetOutputObject()->GetProjectionOffsetsY()[noProj],
                                    geometryReader->GetOutputObject()->GetOutOfPlaneAngles()[noProj],
                                    geometryReader->GetOutputObject()->GetInPlaneAngles()[noProj],
                                    geometryReader->GetOutputObject()->GetSourceOffsetsX()[noProj],
                                    geometryReader->GetOutputObject()->GetSourceOffsetsY()[noProj]);
      outputProjectionsNames.push_back(names->GetFileNames()[noProj]);
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

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Projections reader
  if(args_info.verbose_flag)
    std::cout << "Reading and writing projections in "
              << args_info.out_proj_arg
              << "..."
              << std::endl;
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( outputProjectionsNames );

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.out_proj_arg );
  writer->SetInput( reader->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->UpdateOutputInformation() )
  writer->SetNumberOfStreamDivisions( 1 + reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() / (1024*1024*4) );

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
