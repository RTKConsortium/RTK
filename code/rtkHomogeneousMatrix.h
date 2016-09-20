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

#ifndef rtkHomogeneousMatrix_h
#define rtkHomogeneousMatrix_h

#include <itkMatrix.h>
#include <itkImage.h>

namespace rtk
{

//--------------------------------------------------------------------
/** \brief Get IndexToPhysicalPoint matrix from an image (no accessor provided by ITK).
 *
 * The returned matrix is in homogeneous format.
 *
 * \author Simon Rit
 *
 * \ingroup Functions
 */
//template <class TPixel, unsigned int VImageDimension>
template <class TImageType>
itk::Matrix<double, TImageType::ImageDimension + 1, TImageType::ImageDimension + 1>
GetIndexToPhysicalPointMatrix(const TImageType *image)
{
  const unsigned int Dimension = TImageType::ImageDimension;

  itk::Matrix<double, Dimension + 1, Dimension + 1> matrix;
  matrix.Fill(0.0);

  itk::Index<Dimension>                                 index;
  itk::Point<double, Dimension> point;

  for(unsigned int j=0; j<Dimension; j++)
    {
    index.Fill(0);
    index[j] = 1;
    image->TransformIndexToPhysicalPoint(index,point);
    for(unsigned int i=0; i<Dimension; i++)
      matrix[i][j] = point[i]-image->GetOrigin()[i];
    }
  for(unsigned int i=0; i<Dimension; i++)
    matrix[i][Dimension] = image->GetOrigin()[i];
  matrix[Dimension][Dimension] = 1.0;

  return matrix;
};

//--------------------------------------------------------------------
/** \brief Get PhysicalPointToIndex matrix from an image (no accessor provided by ITK).
 *
 * The returned matrix is in homogeneous format.
 *
 * \author Simon Rit
 *
 * \ingroup Functions
 */
//template <class TPixel, unsigned int VImageDimension>
template <class TImageType>
itk::Matrix<double, TImageType::ImageDimension + 1, TImageType::ImageDimension + 1>
GetPhysicalPointToIndexMatrix(const TImageType *image)
{
  typedef itk::Matrix<double, TImageType::ImageDimension + 1, TImageType::ImageDimension + 1> MatrixType;
  return MatrixType(GetIndexToPhysicalPointMatrix<TImageType>(image).GetInverse() );
};

} // end namespace

#endif // rtkHomogeneousMatrix_h
