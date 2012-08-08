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

#include "rtkthreedshepploganreference_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkDrawSheppLoganFilter.h"
#include <itkImageFileWriter.h>


int main(int argc, char * argv[])
{
  GGO(rtkthreedshepploganreference, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension >  OutputImageType;

  // Empty projection images
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkthreedshepploganreference>(constantImageSource, args_info);

  // Reference
  typedef rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType> DSLType;
  if(args_info.verbose_flag)
    std::cout << "Creating reference... " << std::flush;
  DSLType::Pointer dsl = DSLType::New();
  dsl->SetInput( constantImageSource->GetOutput() );
  dsl->Update();

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( dsl->GetOutput() );
  if(args_info.verbose_flag)
    std::cout << "Writing reference... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
