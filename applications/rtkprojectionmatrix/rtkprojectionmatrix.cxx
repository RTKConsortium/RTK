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

#include <vnl/vnl_sparse_matrix.h>
#include <fstream>

#include "rtkprojectionmatrix_ggo.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkGgoFunctions.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkConfiguration.h"
#include "rtkMatlabSparseMatrix.h"

namespace rtk
{
namespace Functor
{
template <class TInput, class TCoordinateType, class TOutput = TCoordinateType>
class StoreSparseMatrixSplatWeightMultiplication
{
public:
  StoreSparseMatrixSplatWeightMultiplication() = default;
  ~StoreSparseMatrixSplatWeightMultiplication() = default;

  bool
  operator!=(const StoreSparseMatrixSplatWeightMultiplication &) const
  {
    return false;
  }
  bool
  operator==(const StoreSparseMatrixSplatWeightMultiplication & other) const
  {
    return !(*this != other);
  }

  inline void
  operator()(const TInput &        rayValue,
             TOutput &             output,
             const double          stepLengthInVoxel,
             const double          voxelSize,
             const TCoordinateType weight)
  {
    // One row of the matrix is one ray, it should be thread safe
    m_SystemMatrix.put(
      &rayValue - m_ProjectionsBuffer, &output - m_VolumeBuffer, weight * voxelSize * stepLengthInVoxel);
  }
  vnl_sparse_matrix<double> &
  GetVnlSparseMatrix()
  {
    return m_SystemMatrix;
  }
  void
  SetProjectionsBuffer(TInput * pb)
  {
    m_ProjectionsBuffer = pb;
  }
  void
  SetVolumeBuffer(TOutput * vb)
  {
    m_VolumeBuffer = vb;
  }

private:
  vnl_sparse_matrix<double> m_SystemMatrix;
  TInput *                  m_ProjectionsBuffer;
  TOutput *                 m_VolumeBuffer;
};
} // namespace Functor
} // namespace rtk

std::ostream &
operator<<(std::ostream & out, rtk::MatlabSparseMatrix & matlabSparseMatrix)
{
  matlabSparseMatrix.Save(out);
  return out;
}


int
main(int argc, char * argv[])
{
  GGO(rtkprojectionmatrix, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkprojectionmatrix>(reader, args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update())

  // Create back projection image filter
  if (reader->GetOutput()->GetLargestPossibleRegion().GetSize()[1] != 1)
  {
    std::cerr << "This tool has been designed for 2D, i.e., with one row in the sinogram only." << std::endl;
    return EXIT_FAILURE;
  }
  const OutputImageType::DirectionType dir = reader->GetOutput()->GetDirection();
  if (itk::Math::abs(dir[0][0]) != 1. || itk::Math::abs(dir[1][1]) != 1.)
  {
    std::cerr << "Projections with non-diagonal Direction is not handled." << std::endl;
    return EXIT_FAILURE;
  }

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));
  if (args_info.verbose_flag)
    std::cout << " done." << std::endl;

  // Create reconstructed image
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkprojectionmatrix>(constantImageSource,
                                                                                             args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantImageSource->Update())
  if (constantImageSource->GetOutput()->GetLargestPossibleRegion().GetSize()[1] != 3)
  {
    std::cerr << "This tool has been designed for 2D with Joseph project. "
              << "Joseph requires at least 2 slices in the y direction for bilinear interpolation. "
              << "To have one slice exactly in front of the row, use 3 slices in the volume "
              << "with the central slice in front of the single projection row." << std::endl;
    return EXIT_FAILURE;
  }
  if (!constantImageSource->GetOutput()->GetDirection().GetVnlMatrix().is_identity())
  {
    std::cerr << "Volume with non-identity Direction is not handled." << std::endl;
    return EXIT_FAILURE;
  }

  // Adjust size according to geometry
  if (reader->GetOutput()->GetLargestPossibleRegion().GetSize()[2] != geometry->GetGantryAngles().size())
  {
    std::cerr << "Number of projections in the geometry and in the stack do not match." << std::endl;
    return EXIT_FAILURE;
  }

  // Create back projection image filter
  if (args_info.verbose_flag)
    std::cout << "Backprojecting volume and recording matrix values..." << std::endl;

  auto backProjection = rtk::JosephBackProjectionImageFilter<
    OutputImageType,
    OutputImageType,
    rtk::Functor::InterpolationWeightMultiplicationBackProjection<OutputPixelType, OutputPixelType>,
    rtk::Functor::StoreSparseMatrixSplatWeightMultiplication<OutputPixelType, double, OutputPixelType>>::New();
  backProjection->SetInput(constantImageSource->GetOutput());
  backProjection->SetInput(1, reader->GetOutput());
  backProjection->SetGeometry(geometry);
  backProjection->GetSplatWeightMultiplication().SetProjectionsBuffer(reader->GetOutput()->GetBufferPointer());
  backProjection->GetSplatWeightMultiplication().SetVolumeBuffer(constantImageSource->GetOutput()->GetBufferPointer());
  backProjection->GetSplatWeightMultiplication().GetVnlSparseMatrix().resize(
    reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels(),
    constantImageSource->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(backProjection->Update())

  // Write matrix to disk
  if (args_info.verbose_flag)
    std::cout << "Writing matrix to disk..." << std::endl;
  std::ofstream ofs(args_info.output_arg, std::ofstream::binary);
  if (!ofs)
  {
    std::cerr << "Failed to open " << args_info.output_arg << std::endl;
    return EXIT_FAILURE;
  }
  rtk::MatlabSparseMatrix matlabSparseMatrix(backProjection->GetSplatWeightMultiplication().GetVnlSparseMatrix(),
                                             backProjection->GetOutput());
  ofs << matlabSparseMatrix;
  ofs.close();
  return EXIT_SUCCESS;
}
