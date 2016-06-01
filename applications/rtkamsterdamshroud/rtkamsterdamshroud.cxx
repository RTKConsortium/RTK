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

#include "rtkamsterdamshroud_ggo.h"
#include "rtkMacro.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkAmsterdamShroudImageFilter.h"
#include "rtkGgoFunctions.h"

#include <itkImageFileWriter.h>
#include <itkTimeProbe.h>

int main(int argc, char * argv[])
{
  GGO(rtkamsterdamshroud, args_info);

  typedef double OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkamsterdamshroud>(reader, args_info);

  // Amsterdam shroud
  typedef rtk::AmsterdamShroudImageFilter<OutputImageType> ShroudFilterType;
  ShroudFilterType::Pointer shroudFilter = ShroudFilterType::New();
  shroudFilter->SetInput( reader->GetOutput() );
  shroudFilter->SetUnsharpMaskSize(args_info.unsharp_arg);

  // Corners (if given)
  if(args_info.clipbox_given)
    {
    if(args_info.clipbox_given != 6)
      {
      std::cerr << "--clipbox requires 6 values, only"
                << args_info.clipbox_given << " given." << std::endl;
      return EXIT_FAILURE;
      }
    if(!args_info.geometry_given)
      {
      std::cerr << "You must provide the geometry to use --clipbox." << std::endl;
      return EXIT_FAILURE;
      }
    ShroudFilterType::PointType c1, c2;
    for(int i=0; i<3; i++)
      {
      c1[i] = args_info.clipbox_arg[i*2];
      c2[i] = args_info.clipbox_arg[i*2+1];
      }
    shroudFilter->SetCorner1( c1 );
    shroudFilter->SetCorner2( c2 );

    // Geometry
    rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
    geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
    if(args_info.geometry_given)
      {
      geometryReader->SetFilename(args_info.geometry_arg);
      TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() )
      shroudFilter->SetGeometry( geometryReader->GetOutputObject() );
      }
    }
  TRY_AND_EXIT_ON_ITK_EXCEPTION( shroudFilter->UpdateOutputInformation() )

  // Write
  typedef itk::ImageFileWriter< ShroudFilterType::OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( shroudFilter->GetOutput() );
  writer->SetNumberOfStreamDivisions( shroudFilter->GetOutput()->GetLargestPossibleRegion().GetSize(2) );

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
