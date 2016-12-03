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

#include "rtkprojectshepploganphantom_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkAdditiveGaussianNoiseImageFilter.h"

#include <itkImageFileWriter.h>


int main(int argc, char * argv[])
{
  GGO(rtkprojectshepploganphantom, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

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

  // Create a stack of empty projection images
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkprojectshepploganphantom>(constantImageSource, args_info);

  // Adjust size according to geometry
  ConstantImageSourceType::SizeType sizeOutput;
  sizeOutput[0] = constantImageSource->GetSize()[0];
  sizeOutput[1] = constantImageSource->GetSize()[1];
  sizeOutput[2] = geometryReader->GetOutputObject()->GetGantryAngles().size();
  constantImageSource->SetSize( sizeOutput );

  typedef rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  SLPType::Pointer slp=SLPType::New();
  SLPType::VectorType offset(0.);
  SLPType::VectorType scale;
  if(args_info.offset_given)
    {
    offset[0] = args_info.offset_arg[0];
    offset[1] = args_info.offset_arg[1];
    offset[2] = args_info.offset_arg[2];
    }
  scale.Fill(args_info.phantomscale_arg[0]);
  if(args_info.phantomscale_given)
    {
    for(unsigned int i=0; i<vnl_math_min(args_info.phantomscale_given, Dimension); i++)
      scale[i] = args_info.phantomscale_arg[i];
    }
  slp->SetInput(constantImageSource->GetOutput());
  slp->SetGeometry(geometryReader->GetOutputObject());
  slp->SetPhantomScale(scale);
  slp->SetOriginOffset(offset);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( slp->Update() )

  // Add noise
  OutputImageType::Pointer output = slp->GetOutput();
  if(args_info.noise_given)
  {
    typedef rtk::AdditiveGaussianNoiseImageFilter< OutputImageType > NIFType;
    NIFType::Pointer noisy=NIFType::New();
    noisy->SetInput( slp->GetOutput() );
    noisy->SetMean( 0.0 );
    noisy->SetStandardDeviation( args_info.noise_arg );
    TRY_AND_EXIT_ON_ITK_EXCEPTION( noisy->Update() )
    output = noisy->GetOutput();
  }

  // Write
  typedef itk::ImageFileWriter< OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( output );
  if(args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
