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

#include "rtkvectorcgoperator_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkReconstructionConjugateGradientOperator.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkJosephBackProjectionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkTimeProbe.h>
#include <itkVectorImage.h>

int main(int argc, char * argv[])
{
  GGO(rtkvectorcgoperator, args_info);

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
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkvectorcgoperator>(constantImageSource, args_info);
  unsigned int NumberOfChannels = reader->GetOutput()->GetVectorLength();
  constantImageSource->SetVectorLength(NumberOfChannels);

  // Adjust size according to geometry
  ConstantImageSourceType::SizeType sizeOutput;
  sizeOutput[0] = constantImageSource->GetSize()[0];
  sizeOutput[1] = constantImageSource->GetSize()[1];
  sizeOutput[2] = geometryReader->GetOutputObject()->GetGantryAngles().size();
  constantImageSource->SetSize( sizeOutput );

  // Read the block diagonal matrix
  ReaderType::Pointer matrixReader = ReaderType::New();
  matrixReader->SetFileName( args_info.matrix_arg );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( matrixReader->Update() )

  // Create and set the conjugate gradient reconstruction operator
  typedef rtk::ReconstructionConjugateGradientOperator<OutputImageType, itk::Image< OutputPixelType, Dimension > > OperatorType;
  OperatorType::Pointer op = OperatorType::New();
  op->SetInput(0, reader->GetOutput());
  op->SetInput(1, constantImageSource->GetOutput());
  op->SetInput(2, matrixReader->GetOutput());
  op->SetTikhonov(0);
  op->SetGeometry(geometryReader->GetOutputObject());

  typedef rtk::JosephForwardProjectionImageFilter<itk::VectorImage<float, 3>,
              itk::VectorImage<float, 3>,
              rtk::Functor::VectorInterpolationWeightMultiplication<float, double, itk::VariableLengthVector<float>>,
              rtk::Functor::VectorProjectedValueAccumulation<itk::VariableLengthVector<float>, itk::VariableLengthVector<float> > > FWFilterType;
  FWFilterType::Pointer forwardProjectionFilter = FWFilterType::New();
  op->SetForwardProjectionFilter( forwardProjectionFilter.GetPointer() );

  typedef rtk::JosephBackProjectionImageFilter<itk::VectorImage<float, 3>,
              itk::VectorImage<float, 3>,
              rtk::Functor::SplatWeightMultiplication< float, double, float > > BPFilterType;
  BPFilterType::Pointer backProjectionFilter = BPFilterType::New();
  op->SetBackProjectionFilter( backProjectionFilter.GetPointer() );

  TRY_AND_EXIT_ON_ITK_EXCEPTION( op->Update() )

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( op->GetOutput() );

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )
  return EXIT_SUCCESS;
}
