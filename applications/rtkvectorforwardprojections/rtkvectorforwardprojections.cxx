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

#include "rtkvectorforwardprojections_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#ifdef RTK_USE_CUDA
#include "rtkCudaForwardProjectionImageFilter.h"
#endif
#include "rtkRayCastInterpolatorForwardProjectionImageFilter.h"
#include "rtkBlockDiagonalMatrixVectorMultiplyImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkTimeProbe.h>
#include <itkVectorImage.h>

int main(int argc, char * argv[])
{
  GGO(rtkvectorforwardprojections, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef itk::VectorImage< OutputPixelType, Dimension > OutputImageType;
#endif

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


  // Input reader
  if(args_info.verbose_flag)
    std::cout << "Reading input volume "
              << args_info.input_arg
              << "..."
              << std::flush;
  itk::TimeProbe readerProbe;
  typedef itk::ImageFileReader<  OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( args_info.input_arg );
  readerProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )
  readerProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << readerProbe.GetMean() << ' ' << readerProbe.GetUnit()
              << '.' << std::endl;

  // Create a stack of empty projection images
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkvectorforwardprojections>(constantImageSource, args_info);
  unsigned int NumberOfChannels = reader->GetOutput()->GetVectorLength();
  constantImageSource->SetVectorLength(NumberOfChannels);

  // Adjust size according to geometry
  ConstantImageSourceType::SizeType sizeOutput;
  sizeOutput[0] = constantImageSource->GetSize()[0];
  sizeOutput[1] = constantImageSource->GetSize()[1];
  sizeOutput[2] = geometryReader->GetOutputObject()->GetGantryAngles().size();
  constantImageSource->SetSize( sizeOutput );

  // Create forward projection image filter
  if(args_info.verbose_flag)
    std::cout << "Projecting volume..." << std::flush;
  itk::TimeProbe projProbe;

  rtk::ForwardProjectionImageFilter<OutputImageType, OutputImageType>::Pointer forwardProjection;
  
  switch(args_info.fp_arg)
  {
  case(fp_arg_Joseph):
    forwardProjection = rtk::JosephForwardProjectionImageFilter<OutputImageType,
                                                                OutputImageType,
                                                                rtk::Functor::VectorInterpolationWeightMultiplication<typename OutputImageType::InternalPixelType,
                                                                                                                      double,
                                                                                                                      typename OutputImageType::PixelType>,
                                                                rtk::Functor::VectorProjectedValueAccumulation<typename OutputImageType::PixelType,
                                                                                                               typename OutputImageType::PixelType> >::New();
    break;
//  case(fp_arg_RayCastInterpolator):
//    forwardProjection = rtk::RayCastInterpolatorForwardProjectionImageFilter<OutputImageType, OutputImageType>::New();
//    break;
  case(fp_arg_CudaRayCast):
#ifdef RTK_USE_CUDA
    forwardProjection = rtk::CudaForwardProjectionImageFilter<OutputImageType, OutputImageType>::New();
    dynamic_cast<rtk::CudaForwardProjectionImageFilter<OutputImageType, OutputImageType>*>( forwardProjection.GetPointer() )->SetStepSize(args_info.step_arg);
#else
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
#endif
    break;
  default:
    std::cerr << "Unhandled --method value." << std::endl;
    return EXIT_FAILURE;
  }
  forwardProjection->SetInput( constantImageSource->GetOutput() );
  forwardProjection->SetInput( 1, reader->GetOutput() );
  forwardProjection->SetGeometry( geometryReader->GetOutputObject() );
  projProbe.Start();
  if(!args_info.lowmem_flag)
    {
    TRY_AND_EXIT_ON_ITK_EXCEPTION( forwardProjection->Update() )
    }
  projProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << projProbe.GetMean() << ' ' << projProbe.GetUnit()
              << '.' << std::endl;

  // If a block diagonal matrix is given in input, perform multiplication
  typedef rtk::BlockDiagonalMatrixVectorMultiplyImageFilter<OutputImageType> MatrixVectorMultiplyFilterType;
  MatrixVectorMultiplyFilterType::Pointer matrixVectorMultiplyFilter = MatrixVectorMultiplyFilterType::New();
  if (args_info.matrix_given)
    {
    // Read a block diagonal matrix image (typically inverse covariance matrix)
    ReaderType::Pointer matrixReader = ReaderType::New();
    matrixReader->SetFileName( args_info.matrix_arg );
    TRY_AND_EXIT_ON_ITK_EXCEPTION( matrixReader->Update() )

    // Multiply by block diagonal matrix
    matrixVectorMultiplyFilter->SetInput1(forwardProjection->GetOutput()); // First input is the vector
    matrixVectorMultiplyFilter->SetInput2(matrixReader->GetOutput()); // Second input is the matrix
    }

  // Write
  if(args_info.verbose_flag)
    std::cout << "Writing... " << std::flush;
  itk::TimeProbe writeProbe;
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  if (args_info.matrix_given)
    writer->SetInput( matrixVectorMultiplyFilter->GetOutput() );
  else
    writer->SetInput( forwardProjection->GetOutput() );
//  if(args_info.lowmem_flag)
//    {
//    writer->SetNumberOfStreamDivisions(sizeOutput[2]);
//    }
  writeProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )
  writeProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << writeProbe.GetMean() << ' ' << projProbe.GetUnit()
              << '.' << std::endl;

  return EXIT_SUCCESS;
}
