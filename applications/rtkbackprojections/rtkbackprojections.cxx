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

#include "rtkbackprojections_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkProjectionsReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkFDKBackProjectionImageFilter.h"
#include "rtkJosephBackProjectionImageFilter.h"
#if CUDA_FOUND
#  include "rtkCudaFDKBackProjectionImageFilter.h"
#  include "rtkCudaBackProjectionImageFilter.h"
#endif

#include <itkRegularExpressionSeriesFileNames.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkTimeProbe.h>

int main(int argc, char * argv[])
{
  GGO(rtkbackprojections, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Geometry
  if(args_info.verbose_flag)
    std::cout << "Reading geometry information from "
              << args_info.geometry_arg
              << "..."
              << std::flush;
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() )
  if(args_info.verbose_flag)
    std::cout << " done." << std::endl;

  // Create an empty volume
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkbackprojections>(constantImageSource, args_info);

  // Generate file names
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(false);
  names->SetRegularExpression(args_info.regexp_arg);
  names->SetSubMatch(0);

  if(args_info.verbose_flag)
    std::cout << "Reading "
              << names->GetFileNames().size()
              << " projection file(s)..."
              << std::flush;

  // Projections reader
  itk::TimeProbe readerProbe;
  readerProbe.Start();
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() );
  readerProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << readerProbe.GetMeanTime() << ' ' << readerProbe.GetUnit()
              << '.' << std::endl;

  // Create back projection image filter
  if(args_info.verbose_flag)
    std::cout << "Backprojecting volume..." << std::flush;
  itk::TimeProbe bpProbe;
  rtk::BackProjectionImageFilter<OutputImageType, OutputImageType>::Pointer bp;

  switch(args_info.method_arg)
  {
    case(method_arg_VoxelBasedBackProjection):
      bp = rtk::BackProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case(method_arg_FDKBackProjection):
      bp = rtk::FDKBackProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case(method_arg_Joseph):
      bp = rtk::JosephBackProjectionImageFilter<OutputImageType, OutputImageType>::New();
      break;
    case(method_arg_CudaFDKBackProjection):
#if CUDA_FOUND
      bp = rtk::CudaFDKBackProjectionImageFilter::New();
#else
      std::cerr << "The program has not been compiled with cuda option" << std::endl;
      return EXIT_FAILURE;
#endif
      break;
    case(method_arg_CudaBackProjection):
#if CUDA_FOUND
      bp = rtk::CudaBackProjectionImageFilter::New();
#else
      std::cerr << "The program has not been compiled with cuda option" << std::endl;
      return EXIT_FAILURE;
#endif
      break;

    default:
    std::cerr << "Unhandled --method value." << std::endl;
    return EXIT_FAILURE;
  }

  bp->SetInput( constantImageSource->GetOutput() );
  bp->SetInput( 1, reader->GetOutput() );
  bp->SetGeometry( geometryReader->GetOutputObject() );
  bpProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( bp->Update() )
  bpProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << bpProbe.GetMeanTime() << ' ' << bpProbe.GetUnit()
              << '.' << std::endl;

  // Write
  if(args_info.verbose_flag)
    std::cout << "Writing... " << std::flush;
  itk::TimeProbe writeProbe;
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( bp->GetOutput() );
  writeProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );
  writeProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << writeProbe.GetMeanTime() << ' ' << writeProbe.GetUnit()
              << '.' << std::endl;

  return EXIT_SUCCESS;
}
