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

#include "rtkdrawgeometricphantom_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkDrawGeometricPhantomImageFilter.h"
#include "itkImageFileWriter.h"
#include "rtkAdditiveGaussianNoiseImageFilter.h"


int main(int argc, char * argv[])
{
  GGO(rtkdrawgeometricphantom, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension >  OutputImageType;

  // Empty projection images
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkdrawgeometricphantom>(constantImageSource, args_info);

  // Reference
  typedef rtk::DrawGeometricPhantomImageFilter<OutputImageType, OutputImageType> DQType;
  if(args_info.verbose_flag)
    std::cout << "Creating reference... " << std::flush;
  DQType::Pointer dq = DQType::New();
  dq->SetInput( constantImageSource->GetOutput() );
  dq->SetConfigFile(args_info.phantomfile_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dq->Update() )

  // Add noise
  OutputImageType::Pointer output = dq->GetOutput();
  if(args_info.noise_given)
  {
    typedef rtk::AdditiveGaussianNoiseImageFilter< OutputImageType > NIFType;
    NIFType::Pointer noisy=NIFType::New();
    noisy->SetInput( output );
    noisy->SetMean( 0.0 );
    noisy->SetStandardDeviation( args_info.noise_arg );
    TRY_AND_EXIT_ON_ITK_EXCEPTION( noisy->Update() )
    output = noisy->GetOutput();
  }

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( output );
  if(args_info.verbose_flag)
    std::cout << "Writing reference... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
