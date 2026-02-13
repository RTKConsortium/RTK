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

#ifndef rtkMatlabSparseMatrix_hxx
#define rtkMatlabSparseMatrix_hxx

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace rtk
{

//====================================================================
template <class TOutputImage>
MatlabSparseMatrix<TOutputImage>::~MatlabSparseMatrix()
{
  if (m_MatlabSparseMatrix.s_rowIndex)
    delete[] m_MatlabSparseMatrix.s_rowIndex;
  if (m_MatlabSparseMatrix.s_columnIndex)
    delete[] m_MatlabSparseMatrix.s_columnIndex;
  if (m_MatlabSparseMatrix.s_value)
    delete[] m_MatlabSparseMatrix.s_value;
}

template <class TOutputImage>
void
MatlabSparseMatrix<TOutputImage>::BuildMatlabMatrix()
{
  if (!m_Output)
  {
    itkExceptionMacro("Output image not set");
  }

  unsigned int nbColumn = m_Output->GetLargestPossibleRegion().GetSize()[0] *
                          m_Output->GetLargestPossibleRegion().GetSize()[2]; // it's not m_Matrix.columns()

  // Take the number of non-zero elements:
  // Compute the column index
  // Store elements in std::vector and sort them according 1\ index of column and 2\ index of row
  unsigned int nonZeroElement(0);
  using sparseMatrixColumn = std::vector<std::pair<unsigned int, double>>;
  auto * columnsVector = new sparseMatrixColumn[nbColumn];
  m_Matrix.reset();
  while (m_Matrix.next())
  {
    typename TOutputImage::IndexType idx = m_Output->ComputeIndex(m_Matrix.getcolumn());
    if (idx[1] != 1)
      continue;
    unsigned int indexColumn = idx[0] + idx[2] * m_Output->GetLargestPossibleRegion().GetSize()[2];
    auto         it = columnsVector[indexColumn].begin();
    while (it != columnsVector[indexColumn].end())
    {
      if ((unsigned int)m_Matrix.getrow() < it->first)
        break;
      ++it;
    }
    columnsVector[indexColumn].insert(it, std::make_pair(m_Matrix.getrow(), m_Matrix.value()));
    ++nonZeroElement;
  }

  // Store the sparse matrix into a matlab structure
  std::string headerMatlab("MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Fri May 18 14:11:56 2018               "
                           "                        ");
  for (unsigned int i = 0; i < 116; ++i)
    m_MatlabSparseMatrix.s_headerMatlab[i] = headerMatlab[i];
  for (unsigned char & i : m_MatlabSparseMatrix.s_headerOffset)
    i = 0;
  m_MatlabSparseMatrix.s_headerVersion = 0x0100;
  m_MatlabSparseMatrix.s_headerEndian[0] = 'I';
  m_MatlabSparseMatrix.s_headerEndian[1] = 'M';
  m_MatlabSparseMatrix.s_mainTag = 14;
  m_MatlabSparseMatrix.s_arrayTag = 6;
  m_MatlabSparseMatrix.s_arrayLength = 8;
  m_MatlabSparseMatrix.s_arrayFlags = 5;
  m_MatlabSparseMatrix.s_arrayClass = 16;
  m_MatlabSparseMatrix.s_arrayUndefined = 0;
  m_MatlabSparseMatrix.s_arrayNzmax = nonZeroElement;
  m_MatlabSparseMatrix.s_dimensionTag = 5;
  m_MatlabSparseMatrix.s_dimensionLength = 8;
  m_MatlabSparseMatrix.s_dimensionNbRow = m_Matrix.rows();
  m_MatlabSparseMatrix.s_dimensionNbColumn = nbColumn;
  m_MatlabSparseMatrix.s_nameLength = 1;
  m_MatlabSparseMatrix.s_nameTag = 1;
  m_MatlabSparseMatrix.s_nameChar = 'A';
  for (unsigned char & i : m_MatlabSparseMatrix.s_namePadding)
    i = 0;
  m_MatlabSparseMatrix.s_dataLength = 64 + nonZeroElement * 8 + 4 * m_MatlabSparseMatrix.s_arrayNzmax +
                                      4 * (m_MatlabSparseMatrix.s_dimensionNbColumn + 1);
  if (m_MatlabSparseMatrix.s_arrayNzmax * 4 % 8 != 0)
    m_MatlabSparseMatrix.s_dataLength += 4;
  if ((m_MatlabSparseMatrix.s_dimensionNbColumn + 1) * 4 % 8 != 0)
    m_MatlabSparseMatrix.s_dataLength += 4;
  m_MatlabSparseMatrix.s_rowIndexTag = 5;
  m_MatlabSparseMatrix.s_rowIndexLength = 4 * m_MatlabSparseMatrix.s_arrayNzmax; // put it in hexa
  m_MatlabSparseMatrix.s_rowIndex = new unsigned long int[m_MatlabSparseMatrix.s_arrayNzmax];
  m_MatlabSparseMatrix.s_rowIndexPadding = 0;
  m_MatlabSparseMatrix.s_columnIndexTag = 5;
  m_MatlabSparseMatrix.s_columnIndexLength = 4 * (m_MatlabSparseMatrix.s_dimensionNbColumn + 1);
  m_MatlabSparseMatrix.s_columnIndex = new unsigned long int[m_MatlabSparseMatrix.s_dimensionNbColumn + 1];
  m_MatlabSparseMatrix.s_columnIndexPadding = 0;
  m_MatlabSparseMatrix.s_valueTag = 9;
  m_MatlabSparseMatrix.s_valueLength = 8 * m_MatlabSparseMatrix.s_arrayNzmax;
  m_MatlabSparseMatrix.s_value = new double[m_MatlabSparseMatrix.s_arrayNzmax];
  // Copy data
  unsigned int elementIndex(0);
  for (unsigned int i = 0; i < m_MatlabSparseMatrix.s_dimensionNbColumn; ++i)
  {
    m_MatlabSparseMatrix.s_columnIndex[i] = elementIndex;
    for (const auto & it : columnsVector[i])
    {
      m_MatlabSparseMatrix.s_rowIndex[elementIndex] = it.first;
      m_MatlabSparseMatrix.s_value[elementIndex] = it.second;
      ++elementIndex;
    }
  }
  m_MatlabSparseMatrix.s_columnIndex[m_MatlabSparseMatrix.s_dimensionNbColumn] = m_MatlabSparseMatrix.s_arrayNzmax;

  delete[] columnsVector;
}

template <class TOutputImage>
void
MatlabSparseMatrix<TOutputImage>::Save(std::ostream & out)
{
  BuildMatlabMatrix();
  
  // Write data
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_headerMatlab), 116);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_headerOffset), 8);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_headerVersion), 2);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_headerEndian), 2);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_mainTag), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_dataLength), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_arrayTag), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_arrayLength), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_arrayFlags), 1);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_arrayClass), 1);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_arrayUndefined), 2);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_arrayNzmax), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_dimensionTag), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_dimensionLength), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_dimensionNbRow), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_dimensionNbColumn), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_nameLength), 2);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_nameTag), 2);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_nameChar), 1);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_namePadding), 3);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_rowIndexTag), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_rowIndexLength), 4);
  for (unsigned int i = 0; i < m_MatlabSparseMatrix.s_arrayNzmax; ++i)
    out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_rowIndex[i]), 4);
  if (m_MatlabSparseMatrix.s_arrayNzmax * 4 % 8 != 0)
    out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_rowIndexPadding), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_columnIndexTag), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_columnIndexLength), 4);
  for (unsigned int i = 0; i < m_MatlabSparseMatrix.s_dimensionNbColumn + 1; ++i)
    out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_columnIndex[i]), 4);
  if ((m_MatlabSparseMatrix.s_dimensionNbColumn + 1) * 4 % 8 != 0)
    out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_columnIndexPadding), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_valueTag), 4);
  out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_valueLength), 4);
  for (unsigned int i = 0; i < m_MatlabSparseMatrix.s_arrayNzmax; ++i)
    out.write(reinterpret_cast<const char *>(&m_MatlabSparseMatrix.s_value[i]), 8);
}

template <class TOutputImage>
inline void
MatlabSparseMatrix<TOutputImage>::Print()
{
  std::cout << "m_MatlabSparseMatrix.s_headerMatlab : \"" << m_MatlabSparseMatrix.s_headerMatlab << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_headerOffset : \"" << m_MatlabSparseMatrix.s_headerOffset << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_headerVersion : \"" << std::hex << +m_MatlabSparseMatrix.s_headerVersion << "\""
            << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_headerEndian : \"" << m_MatlabSparseMatrix.s_headerEndian << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_mainTag : \"" << m_MatlabSparseMatrix.s_mainTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_dataLength : \"" << m_MatlabSparseMatrix.s_dataLength << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_arrayTag : \"" << m_MatlabSparseMatrix.s_arrayTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_arrayLength : \"" << m_MatlabSparseMatrix.s_arrayLength << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_arrayFlags : \"" << (unsigned int)m_MatlabSparseMatrix.s_arrayFlags << "\""
            << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_arrayClass : \"" << (unsigned int)m_MatlabSparseMatrix.s_arrayClass << "\""
            << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_arrayUndefined : \"" << m_MatlabSparseMatrix.s_arrayUndefined << "\""
            << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_arrayNzmax : \"" << m_MatlabSparseMatrix.s_arrayNzmax << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_dimensionTag : \"" << m_MatlabSparseMatrix.s_dimensionTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_dimensionLength : \"" << m_MatlabSparseMatrix.s_dimensionLength << "\""
            << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_dimensionNbRow : \"" << m_MatlabSparseMatrix.s_dimensionNbRow << "\""
            << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_dimensionNbColumn : \"" << m_MatlabSparseMatrix.s_dimensionNbColumn << "\""
            << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_nameLength : \"" << m_MatlabSparseMatrix.s_nameLength << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_nameTag : \"" << m_MatlabSparseMatrix.s_nameTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_nameChar : \"" << m_MatlabSparseMatrix.s_nameChar << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_namePadding : \"" << m_MatlabSparseMatrix.s_namePadding << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_rowIndexTag : \"" << m_MatlabSparseMatrix.s_rowIndexTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_rowIndexLength : \"" << m_MatlabSparseMatrix.s_rowIndexLength << "\""
            << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_rowIndex : \"";
  for (unsigned int i = 0; i < m_MatlabSparseMatrix.s_arrayNzmax; ++i)
    std::cout << m_MatlabSparseMatrix.s_rowIndex[i] << " ";
  std::cout << "\"" << std::endl;
  if (m_MatlabSparseMatrix.s_arrayNzmax * 4 % 8 != 0)
    std::cout << "m_MatlabSparseMatrix.s_rowIndexPadding : \"" << m_MatlabSparseMatrix.s_rowIndexPadding << "\""
              << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_columnIndexTag : \"" << m_MatlabSparseMatrix.s_columnIndexTag << "\""
            << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_columnIndexLength : \"" << m_MatlabSparseMatrix.s_columnIndexLength << "\""
            << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_columnIndex : \"";
  for (unsigned int i = 0; i < m_MatlabSparseMatrix.s_dimensionNbColumn + 1; ++i)
    std::cout << m_MatlabSparseMatrix.s_columnIndex[i] << " ";
  std::cout << "\"" << std::endl;
  if ((m_MatlabSparseMatrix.s_dimensionNbColumn + 1) * 4 % 8 != 0)
    std::cout << "m_MatlabSparseMatrix.s_columnIndexPadding : \"" << m_MatlabSparseMatrix.s_columnIndexPadding << "\""
              << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_valueTag : \"" << m_MatlabSparseMatrix.s_valueTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_valueLength : \"" << m_MatlabSparseMatrix.s_valueLength << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_value : \"";
  for (unsigned int i = 0; i < m_MatlabSparseMatrix.s_arrayNzmax; ++i)
    std::cout << m_MatlabSparseMatrix.s_value[i] << " ";
  std::cout << "\"" << std::endl;
}

} // namespace rtk

#endif
