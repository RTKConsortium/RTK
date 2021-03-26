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

#include "rtkpilines_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "rtkThreeDHelicalProjectionGeometryXMLFileReader.h"
#include "rtkPILineImageFilter.h"
#include "rtkProgressCommands.h"

#include <itkStreamingImageFilter.h>
#include <itkImageRegionSplitterDirection.h>
#include <itkImageRegionIterator.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkVectorImage.h>

int
main(int argc, char * argv[])
{
  GGO(rtkpilines, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

  using InputImageType = itk::Image<OutputPixelType, Dimension>;
  // using OutputImageType = itk::VectorImage<float,3>;
  using OutputImageType = itk::Image<itk::Vector<OutputPixelType, 2>, Dimension>;

  // Projections reader
  using ReaderType = itk::ImageFileReader<InputImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(args_info.input_arg);

  if (args_info.verbose_flag)
    std::cout << "Reading... " << std::endl;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update())

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDHelicalProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDHelicalProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometryReader->GenerateOutputInformation())
  rtk::ThreeDHelicalProjectionGeometry::Pointer geometry;
  geometry = geometryReader->GetOutputObject();
  geometry->VerifyHelixParameters();


  // Check on hardware parameter
#ifndef RTK_USE_CUDA
  if (!strcmp(args_info.hardware_arg, "cuda"))
  {
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
  }
#endif

  using PILineFilterType = rtk::PILineImageFilter<InputImageType, OutputImageType>;
  PILineFilterType::Pointer pilines = PILineFilterType::New();
  pilines->SetGeometry(geometry);
  pilines->SetInput(reader->GetOutput());
  pilines->Update();

  if (!strcmp(args_info.filetype_arg, "2"))
  {

    InputImageType::RegionType region = reader->GetOutput()->GetLargestPossibleRegion();

    OutputImageType::Pointer outImage = pilines->GetOutput();

    InputImageType::Pointer im1 = InputImageType::New();
    im1->CopyInformation(reader->GetOutput());
    im1->SetRegions(region);
    im1->Allocate();

    InputImageType::Pointer im2 = InputImageType::New();
    im2->CopyInformation(reader->GetOutput());
    im2->SetRegions(region);
    im2->Allocate();

    using RegionIterator = itk::ImageRegionIterator<InputImageType>;
    using OutRegionIterator = itk::ImageRegionIterator<OutputImageType>;

    RegionIterator    it1(im1, im1->GetLargestPossibleRegion());
    RegionIterator    it2(im2, im2->GetLargestPossibleRegion());
    OutRegionIterator out(pilines->GetOutput(), pilines->GetOutput()->GetLargestPossibleRegion());

    for (it1.GoToBegin(), it2.GoToBegin(), out.GoToBegin(); !it1.IsAtEnd(); ++it1, ++it2, ++out)
    {
      using VectorType = itk::Vector<OutputPixelType, 2>;
      VectorType vector = out.Get();

      it1.Set(vector[0]);
      it2.Set(vector[1]);
    }

    // Write
    using WriterType = itk::ImageFileWriter<InputImageType>;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(args_info.output1_arg);
    writer->SetInput(im1);

    if (args_info.verbose_flag)
      std::cout << "Reconstructing and writing... " << std::endl;

    TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

    writer->SetFileName(args_info.output2_arg);
    writer->SetInput(im2);

    TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())
  }
  else if (!strcmp(args_info.filetype_arg, "1"))
  {
    // Write
    using WriterType = itk::ImageFileWriter<OutputImageType>;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(args_info.output1_arg);
    writer->SetInput(pilines->GetOutput());

    if (args_info.verbose_flag)
      std::cout << "Reconstructing and writing... " << std::endl;

    TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())
  }

  else
  {
    std::cerr << "The filetype option was not properly set." << std::endl;
    std::cerr << "1 : one vector image file. 2 : 2 single valued images." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
