/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtksubselect_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkProjectionsReader.h"
#include "rtkConstantImageSource.h"

#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkPasteImageFilter.h>

int
main(int argc, char * argv[])
{
  GGO(rtksubselect, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtksubselect>(reader, args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update())

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

  // Compute the indices of the selected projections
  std::vector<int> indices;
  int              n = geometry->GetGantryAngles().size();
  if (args_info.last_given)
    n = std::min(args_info.last_arg, n);
  if (args_info.list_given)
    for (unsigned int i = 0; i < args_info.list_given; i++)
    {
      indices.push_back(args_info.list_arg[i]);
    }
  else
    for (int noProj = args_info.first_arg; noProj < n; noProj += args_info.step_arg)
    {
      indices.push_back(noProj);
    }

  // Output RTK geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto outputGeometry = GeometryType::New();

  // Output projections object
  using SourceType = rtk::ConstantImageSource<OutputImageType>;
  auto source = SourceType::New();
  source->SetInformationFromImage(reader->GetOutput());
  OutputImageType::SizeType outputSize = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
  outputSize[Dimension - 1] = indices.size();
  source->SetSize(outputSize);
  source->SetConstant(0);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(source->Update())

  // Fill in the outputGeometry and the output projections
  using PasteType = itk::PasteImageFilter<OutputImageType>;
  auto paste = PasteType::New();
  paste->SetSourceImage(reader->GetOutput());
  paste->SetDestinationImage(source->GetOutput());

  OutputImageType::RegionType sourceRegion;
  OutputImageType::IndexType  destinationIndex;
  for (unsigned int i = 0; i < indices.size(); i++)
  {
    // If it is not the first projection, we need to use the output of
    // the paste filter as input
    if (i)
    {
      OutputImageType::Pointer pimg = paste->GetOutput();
      pimg->DisconnectPipeline();
      paste->SetDestinationImage(pimg);
    }

    sourceRegion = reader->GetOutput()->GetLargestPossibleRegion();
    sourceRegion.SetIndex(Dimension - 1, indices[i]);
    sourceRegion.SetSize(Dimension - 1, 1);
    paste->SetSourceRegion(sourceRegion);

    destinationIndex = reader->GetOutput()->GetLargestPossibleRegion().GetIndex();
    destinationIndex[Dimension - 1] = i;
    paste->SetDestinationIndex(destinationIndex);

    TRY_AND_EXIT_ON_ITK_EXCEPTION(paste->Update())

    // Fill in the output geometry object
    outputGeometry->SetRadiusCylindricalDetector(geometry->GetRadiusCylindricalDetector());
    outputGeometry->AddProjectionInRadians(geometry->GetSourceToIsocenterDistances()[indices[i]],
                                           geometry->GetSourceToDetectorDistances()[indices[i]],
                                           geometry->GetGantryAngles()[indices[i]],
                                           geometry->GetProjectionOffsetsX()[indices[i]],
                                           geometry->GetProjectionOffsetsY()[indices[i]],
                                           geometry->GetOutOfPlaneAngles()[indices[i]],
                                           geometry->GetInPlaneAngles()[indices[i]],
                                           geometry->GetSourceOffsetsX()[indices[i]],
                                           geometry->GetSourceOffsetsY()[indices[i]]);
    outputGeometry->SetCollimationOfLastProjection(geometry->GetCollimationUInf()[indices[i]],
                                                   geometry->GetCollimationUSup()[indices[i]],
                                                   geometry->GetCollimationVInf()[indices[i]],
                                                   geometry->GetCollimationVSup()[indices[i]]);
  }

  // Geometry writer
  if (args_info.verbose_flag)
    std::cout << "Writing geometry information in " << args_info.out_geometry_arg << "..." << std::endl;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(rtk::WriteGeometry(outputGeometry, args_info.out_geometry_arg))

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(paste->GetOutput(), args_info.out_proj_arg))

  return EXIT_SUCCESS;
}
