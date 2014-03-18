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

#include "rtkfieldofview_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkFieldOfViewImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>


int main(int argc, char * argv[])
{
  GGO(rtkfieldofview, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkfieldofview>(reader, args_info);

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

  // Reconstruction reader
  typedef itk::ImageFileReader<  OutputImageType > ImageReaderType;
  ImageReaderType::Pointer unmasked_reconstruction = ImageReaderType::New();
  unmasked_reconstruction->SetFileName(args_info.reconstruction_arg);

  // FOV filter
  typedef rtk::FieldOfViewImageFilter<OutputImageType, OutputImageType> FOVFilterType;
  FOVFilterType::Pointer fieldofview=FOVFilterType::New();
  fieldofview->SetMask(args_info.mask_flag);
  fieldofview->SetInput(0, unmasked_reconstruction->GetOutput());
  fieldofview->SetProjectionsStack(reader->GetOutput());
  fieldofview->SetGeometry(geometryReader->GetOutputObject());
  fieldofview->SetDisplacedDetector(args_info.displaced_flag);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fieldofview->Update() );

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( fieldofview->GetOutput() );
  if(args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
