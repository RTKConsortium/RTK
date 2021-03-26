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

#include "rtkkatsevichtrash_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "rtkThreeDHelicalProjectionGeometryXMLFileReader.h"
#include "rtkProgressCommands.h"
#include "rtkFFTHilbertImageFilter.h"
#include <rtkPILineImageFilter.h>


#include <itkStreamingImageFilter.h>
#include <itkImageRegionSplitterDirection.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkkatsevichtrash, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

  using CPUOutputImageType = itk::Image<OutputPixelType, Dimension>;
#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = CPUOutputImageType;
#endif

  using PILineRealType = float;
  using PILineImageType = itk::Image<itk::Vector<PILineRealType, 2>, Dimension>;
  using PILineImagePointer = typename PILineImageType::Pointer;
  using PILineImageFilterType = rtk::PILineImageFilter<OutputImageType, PILineImageType>;
  using PILinePointer = typename PILineImageFilterType::Pointer;


  //// Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkkatsevichtrash>(reader, args_info);

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

  OutputImageType::Pointer     cst = OutputImageType::New();
  OutputImageType::RegionType  region;
  OutputImageType::IndexType   index;
  OutputImageType::SizeType    size;
  OutputImageType::SpacingType spacing;
  OutputImageType::PointType   origin;
  spacing.Fill(1.);
  index.Fill(0);
  size[0] = 256;
  size[1] = 1;
  size[2] = 256;
  origin[0] = -0.5 * 255;
  origin[1] = 0.;
  origin[2] = -0.5 * 255;
  region.SetSize(size);
  region.SetIndex(index);
  cst->SetRegions(region);
  cst->SetSpacing(spacing);
  cst->SetOrigin(origin);
  cst->Allocate();
  cst->FillBuffer(0.);

  PILineImageFilterType::Pointer pil = PILineImageFilterType::New();
  pil->SetInput(cst);
  pil->SetGeometry(geometry);
  pil->Update();


  OutputImageType::Pointer pil1 = OutputImageType::New();
  OutputImageType::Pointer pil2 = OutputImageType::New();
  pil1->SetRegions(region);
  pil1->SetSpacing(spacing);
  pil1->SetOrigin(origin);
  pil1->Allocate();
  pil2->SetRegions(region);
  pil2->SetSpacing(spacing);
  pil2->SetOrigin(origin);
  pil2->Allocate();

  using PILineIterator = itk::ImageRegionConstIterator<PILineImageType>;
  using OutputRegionIterator = itk::ImageRegionIteratorWithIndex<OutputImageType>;


  PILineIterator itPIL(pil->GetOutput(),region);
  OutputRegionIterator itPIL1(pil1,region);
  OutputRegionIterator itPIL2(pil2,region);

  itPIL.GoToBegin();
  itPIL1.GoToBegin();
  itPIL2.GoToBegin();
  itk::Vector<PILineRealType,2> pil_bounds;

  while (!itPIL.IsAtEnd())
  {
    pil_bounds = itPIL.Get();
    itPIL1.Set(pil_bounds[0]);
    itPIL2.Set(pil_bounds[1]);
    ++itPIL;
    ++itPIL1;
    ++itPIL2;
  }


  // Check on hardware parameter
#ifndef RTK_USE_CUDA
  if (!strcmp(args_info.hardware_arg, "cuda"))
  {
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
  }
#endif


  // Write
  using WriterType = itk::ImageFileWriter<CPUOutputImageType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(pil1);

  if (args_info.verbose_flag)
    std::cout << "Reconstructing and writing... " << std::endl;

  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  writer->SetFileName(args_info.input_arg);
  writer->SetInput(pil2);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())


  return EXIT_SUCCESS;
}
