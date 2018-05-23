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

#ifndef rtkMatlabSparseMatrix_h
#define rtkMatlabSparseMatrix_h

#include <vnl/vnl_sparse_matrix.h>

namespace rtk
{
/** \class MatlabSparseMatrix
 *
 * sparse matrix in Matlab format
 * Initilaize it with vnl_sparse_matrix
 * Save it into .mat format
 *
 */

class MatlabSparseMatrix
{
public:
  template <class TOutputImage> MatlabSparseMatrix(const vnl_sparse_matrix<double>& sparseMatrix, TOutputImage* output);
  void Save(std::ostream& out);
  void Print();

  struct MatlabSparseMatrixStruct {
    unsigned char s_headerMatlab[116];
    unsigned char s_headerOffset[8];
    unsigned short int s_headerVersion;
    unsigned char s_headerEndian[2];
    unsigned long int s_mainTag;
    unsigned long int s_dataLength;
    unsigned long int s_arrayTag;
    unsigned long int s_arrayLength;
    unsigned short int s_arrayUndefined;
    unsigned char s_arrayFlags;
    unsigned char s_arrayClass;
    unsigned long int s_arrayNzmax;
    unsigned long int s_dimensionTag;
    unsigned long int s_dimensionLength;
    unsigned long int s_dimensionNbRow;
    unsigned long int s_dimensionNbColumn;
    unsigned short int s_nameLength;
    unsigned short int s_nameTag;
    unsigned char s_nameChar;
    unsigned char s_namePadding[3];
    unsigned long int s_rowIndexTag;
    unsigned long int s_rowIndexLength;
    unsigned long int *s_rowIndex;
    unsigned long int s_rowIndexPadding;
    unsigned long int s_columnIndexTag;
    unsigned long int s_columnIndexLength;
    unsigned long int *s_columnIndex;
    unsigned long int s_columnIndexPadding;
    unsigned long int s_valueTag;
    unsigned long int s_valueLength;
    double *s_value;
  };

protected:
  MatlabSparseMatrixStruct m_MatlabSparseMatrix;

};
} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkMatlabSparseMatrix.hxx"
#endif

#endif