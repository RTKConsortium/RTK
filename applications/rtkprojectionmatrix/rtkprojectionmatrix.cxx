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

#include <vnl/vnl_sparse_matrix.h>
#include <fstream>

#include "rtkprojectionmatrix_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkConfiguration.h"

namespace rtk
{
namespace Functor
{
template< class TInput, class TCoordRepType, class TOutput=TCoordRepType >
class StoreSparseMatrixSplatWeightMultiplication
{
public:
  StoreSparseMatrixSplatWeightMultiplication() {};
  ~StoreSparseMatrixSplatWeightMultiplication() {};
  bool operator!=( const StoreSparseMatrixSplatWeightMultiplication& ) const
    {
    return false;
    }
  bool operator==(const StoreSparseMatrixSplatWeightMultiplication& other) const
    {
    return !( *this != other );
    }

  inline void operator()( const TInput &rayValue,
                          TOutput &output,
                          const double stepLengthInVoxel,
                          const double voxelSize,
                          const TCoordRepType weight)
    {
    // One row of the matrix is one ray, it should be thread safe
    m_SystemMatrix.put(&rayValue - m_ProjectionsBuffer,
                       &output - m_VolumeBuffer,
                       weight*voxelSize*stepLengthInVoxel);
    }
  vnl_sparse_matrix<double> &GetVnlSparseMatrix() { return m_SystemMatrix; }
  void SetProjectionsBuffer(TInput *pb) { m_ProjectionsBuffer = pb; }
  void SetVolumeBuffer(TOutput *vb) { m_VolumeBuffer = vb; }

private:
  vnl_sparse_matrix<double> m_SystemMatrix;
  TInput *                  m_ProjectionsBuffer;
  TOutput *                 m_VolumeBuffer;
};
}
}

int main(int argc, char * argv[])
{
  GGO(rtkprojectionmatrix, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkprojectionmatrix>(reader, args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )

  // Create back projection image filter
  if(reader->GetOutput()->GetLargestPossibleRegion().GetSize()[1]!=1)
    {
    std::cerr << "This tool has been designed for 2D, i.e., with one row in the sinogram only." << std::endl;
    return EXIT_FAILURE;
    }
  if(!reader->GetOutput()->GetDirection().GetVnlMatrix().is_identity ())
    {
    std::cerr << "Projections with non-identity Direction is not handled." << std::endl;
    return EXIT_FAILURE;
    }

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
  if(args_info.verbose_flag)
    std::cout << " done." << std::endl;

  // Create reconstructed image
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkprojectionmatrix>(constantImageSource, args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( constantImageSource->Update() )
  if(constantImageSource->GetOutput()->GetLargestPossibleRegion().GetSize()[1]!=3)
    {
    std::cerr << "This tool has been designed for 2D with Joseph project. "
              << "Joseph requires at least 2 slices in the y direction for bilinear interpolation. "
              << "To have one slice exactly in front of the row, use 3 slices in the volume "
              << "with the central slice in front of the single projection row."
              << std::endl;
    return EXIT_FAILURE;
    }
  if(!constantImageSource->GetOutput()->GetDirection().GetVnlMatrix().is_identity ())
    {
    std::cerr << "Volume with non-identity Direction is not handled." << std::endl;
    return EXIT_FAILURE;
    }

  // Adjust size according to geometry
  if( reader->GetOutput()->GetLargestPossibleRegion().GetSize()[2] != geometryReader->GetOutputObject()->GetGantryAngles().size() )
    {
    std::cerr << "Number of projections in the geometry and in the stack do not match." << std::endl;
    return EXIT_FAILURE;
    }

  // Create back projection image filter
  if(args_info.verbose_flag)
    std::cout << "Backprojecting volume and recording matrix values..." << std::endl;

  typedef rtk::JosephBackProjectionImageFilter<OutputImageType,
                                               OutputImageType,
                                               rtk::Functor::StoreSparseMatrixSplatWeightMultiplication<OutputPixelType, double, OutputPixelType> >
                                                 JosephType;
  JosephType::Pointer backProjection = JosephType::New();
  backProjection->SetInput( constantImageSource->GetOutput() );
  backProjection->SetInput( 1, reader->GetOutput() );
  backProjection->SetGeometry( geometryReader->GetOutputObject() );
  backProjection->GetSplatWeightMultiplication().SetProjectionsBuffer( reader->GetOutput()->GetBufferPointer() );
  backProjection->GetSplatWeightMultiplication().SetVolumeBuffer( constantImageSource->GetOutput()->GetBufferPointer() );
  backProjection->GetSplatWeightMultiplication().GetVnlSparseMatrix().resize(
          reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels(),
          constantImageSource->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels());
  TRY_AND_EXIT_ON_ITK_EXCEPTION( backProjection->Update() )

  // Write matrix to disk
  if(args_info.verbose_flag)
    std::cout << "Writing matrix to disk..." << std::endl;
  std::ofstream ofs(args_info.output_arg, std::ofstream::binary);
  if(!ofs)
    {
    std::cerr << "Failed to open " << args_info.output_arg << std::endl;
    return EXIT_FAILURE;
    }
  backProjection->GetSplatWeightMultiplication().GetVnlSparseMatrix().reset();
  double element[3];
  while(backProjection->GetSplatWeightMultiplication().GetVnlSparseMatrix().next())
    {
    element[0] = (double)backProjection->GetSplatWeightMultiplication().GetVnlSparseMatrix().getrow();
    int col = (double)backProjection->GetSplatWeightMultiplication().GetVnlSparseMatrix().getcolumn();
    OutputImageType::IndexType idx = backProjection->GetOutput()->ComputeIndex(col);
    if(idx[1] != 1)
      continue;
    element[1] = idx[0] + idx[2]*backProjection->GetOutput()->GetLargestPossibleRegion().GetSize()[2];
    element[2] = (double)backProjection->GetSplatWeightMultiplication().GetVnlSparseMatrix().value();
    ofs.write((char *)element, sizeof(double)*3);
    }
  // Add a 0. in the file for the last element
  element[0] = reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() - 1;
  element[1] = backProjection->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels()/
               backProjection->GetOutput()->GetLargestPossibleRegion().GetSize()[1] - 1;
  element[2] = 0.;
  ofs.write((char *)element, sizeof(double)*3);
  ofs.close();
  return EXIT_SUCCESS;
}
