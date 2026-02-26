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

#ifndef rtkStoreSparseMatrixSplatWeightMultiplication_h
#define rtkStoreSparseMatrixSplatWeightMultiplication_h

#include <vnl/vnl_sparse_matrix.h>
#include "rtkConfiguration.h"

namespace rtk
{
namespace Functor
{
/**
 * \class StoreSparseMatrixSplatWeightMultiplication
 *
 * \brief Functor to capture and store the back-projection system matrix.
 *
 * This functor is used with JosephBackProjectionImageFilter to capture
 * the sparse system matrix entries during back-projection computation.
 * Each matrix entry A[i,j] represents the contribution of volume voxel j
 * to projection pixel i during the back-projection operation.
 *
 * The matrix entry is computed as: weight * voxelSize * stepLengthInVoxel
 *
 * \ingroup RTK
 */
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

  /**
   * \brief Store matrix entry for this voxel-ray intersection.
   *
   * \param rayValue Pointer to projection pixel value
   * \param output Pointer to volume voxel value
   * \param stepLengthInVoxel Distance traveled through voxel
   * \param voxelSize Physical size of voxel
   * \param weight Interpolation weight from back-projection filter
   */
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

  /**
   * \brief Get reference to the sparse matrix.
   */
  vnl_sparse_matrix<double> &
  GetVnlSparseMatrix()
  {
    return m_SystemMatrix;
  }

  /**
   * \brief Set pointer to projection data buffer for index computation.
   */
  void
  SetProjectionsBuffer(TInput * pb)
  {
    m_ProjectionsBuffer = pb;
  }

  /**
   * \brief Set pointer to volume data buffer for index computation.
   */
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

#endif
