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

#include "rtkrayquadricintersection_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayQuadricIntersectionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkrayquadricintersection, args_info);

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
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkrayquadricintersection>(constantImageSource, args_info);

  // Adjust size according to geometry
  ConstantImageSourceType::SizeType sizeOutput;
  sizeOutput[0] = constantImageSource->GetSize()[0];
  sizeOutput[1] = constantImageSource->GetSize()[1];
  sizeOutput[2] = geometryReader->GetOutputObject()->GetGantryAngles().size();
  constantImageSource->SetSize( sizeOutput );

  // Create projection image filter
  typedef rtk::RayQuadricIntersectionImageFilter<OutputImageType, OutputImageType> RBIType;
  RBIType::Pointer rbi = RBIType::New();
  rbi->SetInput( constantImageSource->GetOutput() );
  if(args_info.parameters_given>0) rbi->GetRQIFunctor()->SetA(args_info.parameters_arg[0]);
  if(args_info.parameters_given>1) rbi->GetRQIFunctor()->SetB(args_info.parameters_arg[1]);
  if(args_info.parameters_given>2) rbi->GetRQIFunctor()->SetC(args_info.parameters_arg[2]);
  if(args_info.parameters_given>3) rbi->GetRQIFunctor()->SetD(args_info.parameters_arg[3]);
  if(args_info.parameters_given>4) rbi->GetRQIFunctor()->SetE(args_info.parameters_arg[4]);
  if(args_info.parameters_given>5) rbi->GetRQIFunctor()->SetF(args_info.parameters_arg[5]);
  if(args_info.parameters_given>6) rbi->GetRQIFunctor()->SetG(args_info.parameters_arg[6]);
  if(args_info.parameters_given>7) rbi->GetRQIFunctor()->SetH(args_info.parameters_arg[7]);
  if(args_info.parameters_given>8) rbi->GetRQIFunctor()->SetI(args_info.parameters_arg[8]);
  if(args_info.parameters_given>9) rbi->GetRQIFunctor()->SetJ(args_info.parameters_arg[9]);
  rbi->SetDensity(args_info.mult_arg);
  rbi->SetGeometry( geometryReader->GetOutputObject() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( rbi->Update() )

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( rbi->GetOutput() );
  if(args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
