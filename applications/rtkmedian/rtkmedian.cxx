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

#include "rtkmedian_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkMedianImageFilter.h"
#include "rtkProjectionsReader.h"
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkTimeProbe.h>


int main(int argc, char * argv[])
{
  GGO(rtkmedian, args_info);

  typedef unsigned short OutputPixelType;
  const unsigned int     Dimension = 2;
  unsigned int           medianWindow[2];

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Input reader
  if(args_info.verbose_flag)
    std::cout << "Reading input volume "
              << args_info.input_arg
              << "..."
              << std::flush;
  itk::TimeProbe readerProbe;
  typedef itk::ImageFileReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( args_info.input_arg );
  readerProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )
  readerProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << readerProbe.GetMean() << ' ' << readerProbe.GetUnit()
              << '.' << std::endl;

  // Reading median Window
  if(args_info.median_given<Dimension)
  {
    for(unsigned int i=0; i<Dimension; i++)
      medianWindow[i] = args_info.median_arg[0];
  }
  else
    for(unsigned int i=0; i<Dimension; i++)
      medianWindow[i] = args_info.median_arg[i];

  // Median filter
  typedef rtk::MedianImageFilter MEDFilterType;
  MEDFilterType::Pointer median=MEDFilterType::New();
  median->SetInput(reader->GetOutput());
  median->SetMedianWindow(medianWindow);
  median->Update();
  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( median->GetOutput() );
  if(args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  writer->SetFileName( "orginal.mha" );
  writer->SetInput( reader->GetOutput() );
  if(args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );


  return EXIT_SUCCESS;
}
