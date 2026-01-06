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

#ifndef rtkMatlabSparseMatrix_h
#define rtkMatlabSparseMatrix_h

#include "RTKExport.h"
#include "rtkConfiguration.h"

#include <itkObject.h>
#include <itkSmartPointer.h>

#include <vnl/vnl_sparse_matrix.h>

#include <ostream>
#include <string>

namespace rtk
{

/** \class MatlabSparseMatrix
 *
 * \brief Sparse matrix in Matlab format
 *
 * Converts a vnl_sparse_matrix to Matlab .mat format.
 * Initialize it with vnl_sparse_matrix and save it into .mat format.
 *
 * \author Thomas Baudier
 *
 * \ingroup RTK
 */
template <typename TOutputImage>
class ITK_TEMPLATE_EXPORT MatlabSparseMatrix : public itk::Object
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MatlabSparseMatrix);

  using Self = MatlabSparseMatrix<TOutputImage>;
  using Superclass = itk::Object;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using OutputImageType = TOutputImage;
  using MatrixType = vnl_sparse_matrix<double>;

  itkOverrideGetNameOfClassMacro(MatlabSparseMatrix);
  itkFactorylessNewMacro(Self);

  // Custom setter needed because vnl_sparse_matrix doesn't support operator<< for itkSetMacro
  void
  SetMatrix(const MatrixType & matrix)
  {
    m_Matrix = matrix;
    this->Modified();
  }
  itkGetMacro(Matrix, MatrixType);

  itkSetConstObjectMacro(Output, OutputImageType);
  itkGetConstObjectMacro(Output, OutputImageType);

  void
  Save(std::ostream & out);

  void
  Print();

protected:
  MatlabSparseMatrix() = default;
  ~MatlabSparseMatrix() override;

  struct MatlabSparseMatrixStruct
  {
    unsigned char       s_headerMatlab[116];
    unsigned char       s_headerOffset[8];
    unsigned short int  s_headerVersion;
    unsigned char       s_headerEndian[2];
    unsigned long int   s_mainTag;
    unsigned long int   s_dataLength;
    unsigned long int   s_arrayTag;
    unsigned long int   s_arrayLength;
    unsigned short int  s_arrayUndefined;
    unsigned char       s_arrayFlags;
    unsigned char       s_arrayClass;
    unsigned long int   s_arrayNzmax;
    unsigned long int   s_dimensionTag;
    unsigned long int   s_dimensionLength;
    unsigned long int   s_dimensionNbRow;
    unsigned long int   s_dimensionNbColumn;
    unsigned short int  s_nameLength;
    unsigned short int  s_nameTag;
    unsigned char       s_nameChar;
    unsigned char       s_namePadding[3];
    unsigned long int   s_rowIndexTag;
    unsigned long int   s_rowIndexLength;
    unsigned long int * s_rowIndex = nullptr;
    unsigned long int   s_rowIndexPadding;
    unsigned long int   s_columnIndexTag;
    unsigned long int   s_columnIndexLength;
    unsigned long int * s_columnIndex = nullptr;
    unsigned long int   s_columnIndexPadding;
    unsigned long int   s_valueTag;
    unsigned long int   s_valueLength;
    double *            s_value = nullptr;
  };

private:
  void
  BuildMatlabMatrix();

  MatlabSparseMatrixStruct                       m_MatlabSparseMatrix;
  MatrixType                                     m_Matrix{};
  typename OutputImageType::ConstPointer         m_Output;
};

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkMatlabSparseMatrix.hxx"
#endif

#endif
