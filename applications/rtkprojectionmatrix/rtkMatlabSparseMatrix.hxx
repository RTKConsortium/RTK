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

#include <fstream>
#include <iostream>

//====================================================================
template <class TOutputImage>
rtk::MatlabSparseMatrix::MatlabSparseMatrix(const vnl_sparse_matrix<double>& sparseMatrix, TOutputImage* output)
{
  unsigned int nbColumn = output->GetLargestPossibleRegion().GetSize()[0]*output->GetLargestPossibleRegion().GetSize()[2]; // it's not sparseMatrix.columns()

  //Take the number of non-zero elements:
  //Compute the column index
  //Store elements in std::vector and sort them according 1\ index of column and 2\ index of row
  unsigned int nonZeroElement(0);
  typedef std::vector<std::pair<unsigned int, double> > sparseMatrixColumn;
  sparseMatrixColumn* columnsVector = new sparseMatrixColumn[nbColumn];
  sparseMatrix.reset();
  while(sparseMatrix.next()) {
    typename TOutputImage::IndexType idx = output->ComputeIndex(sparseMatrix.getcolumn());
    if(idx[1] != 1)
      continue;
    unsigned int indexColumn = idx[0] + idx[2]*output->GetLargestPossibleRegion().GetSize()[2];
    sparseMatrixColumn::iterator it = columnsVector[indexColumn].begin();
    while (it != columnsVector[indexColumn].end()) {
      if (sparseMatrix.getrow() < it->first)
        break;
      ++it;
    }
    columnsVector[indexColumn].insert(it, std::make_pair(sparseMatrix.getrow(),sparseMatrix.value()));
    ++nonZeroElement;
  }

  //Store the sparse matrix into a matlab structure
  std::string headerMatlab("MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Fri May 18 14:11:56 2018                                       ");
  for (unsigned int i=0; i<116; ++i)
    m_MatlabSparseMatrix.s_headerMatlab[i] = headerMatlab[i];
  for (unsigned int i=0; i<8; ++i)
    m_MatlabSparseMatrix.s_headerOffset[i] = 0;
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
  m_MatlabSparseMatrix.s_dimensionNbRow = sparseMatrix.rows();
  m_MatlabSparseMatrix.s_dimensionNbColumn = nbColumn;
  m_MatlabSparseMatrix.s_nameLength = 1;
  m_MatlabSparseMatrix.s_nameTag = 1;
  m_MatlabSparseMatrix.s_nameChar = 'A';
  for (unsigned int i=0; i<3; ++i)
    m_MatlabSparseMatrix.s_namePadding[i] = 0;
  m_MatlabSparseMatrix.s_dataLength = 64 + nonZeroElement*8 + 4*m_MatlabSparseMatrix.s_arrayNzmax + 4*(m_MatlabSparseMatrix.s_dimensionNbColumn+1);
  if (m_MatlabSparseMatrix.s_arrayNzmax*4%8 != 0)
    m_MatlabSparseMatrix.s_dataLength += 4;
  if ((m_MatlabSparseMatrix.s_dimensionNbColumn+1)*4%8 != 0)
    m_MatlabSparseMatrix.s_dataLength += 4;
  m_MatlabSparseMatrix.s_rowIndexTag = 5;
  m_MatlabSparseMatrix.s_rowIndexLength = 4*m_MatlabSparseMatrix.s_arrayNzmax; //put it in hexa
  m_MatlabSparseMatrix.s_rowIndex = new unsigned long int[m_MatlabSparseMatrix.s_arrayNzmax];
  m_MatlabSparseMatrix.s_rowIndexPadding = 0;
  m_MatlabSparseMatrix.s_columnIndexTag = 5;
  m_MatlabSparseMatrix.s_columnIndexLength = 4*(m_MatlabSparseMatrix.s_dimensionNbColumn+1);
  m_MatlabSparseMatrix.s_columnIndex = new unsigned long int[m_MatlabSparseMatrix.s_dimensionNbColumn+1];
  m_MatlabSparseMatrix.s_columnIndexPadding = 0;
  m_MatlabSparseMatrix.s_valueTag = 9;
  m_MatlabSparseMatrix.s_valueLength = 8*m_MatlabSparseMatrix.s_arrayNzmax;
  m_MatlabSparseMatrix.s_value = new double[m_MatlabSparseMatrix.s_arrayNzmax];
  //Copy data
  unsigned int elementIndex(0);
  for (unsigned int i=0; i<m_MatlabSparseMatrix.s_dimensionNbColumn; ++i) {
    m_MatlabSparseMatrix.s_columnIndex[i] = elementIndex;
    for (sparseMatrixColumn::iterator it = columnsVector[i].begin(); it != columnsVector[i].end(); ++it) {
      m_MatlabSparseMatrix.s_rowIndex[elementIndex] = it->first;
      m_MatlabSparseMatrix.s_value[elementIndex] = it->second;
      ++elementIndex;
    }
  }
  m_MatlabSparseMatrix.s_columnIndex[m_MatlabSparseMatrix.s_dimensionNbColumn] = m_MatlabSparseMatrix.s_arrayNzmax;
}

void rtk::MatlabSparseMatrix::Save(std::ostream& out) {
  //Write data
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_headerMatlab), 116);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_headerOffset), 8);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_headerVersion), 2);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_headerEndian), 2);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_mainTag), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_dataLength), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_arrayTag), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_arrayLength), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_arrayFlags), 1);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_arrayClass), 1);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_arrayUndefined), 2);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_arrayNzmax), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_dimensionTag), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_dimensionLength), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_dimensionNbRow), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_dimensionNbColumn), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_nameLength), 2);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_nameTag), 2);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_nameChar), 1);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_namePadding), 3);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_rowIndexTag), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_rowIndexLength), 4);
  for (unsigned int i=0; i<m_MatlabSparseMatrix.s_arrayNzmax; ++i)
    out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_rowIndex[i]), 4);
  if (m_MatlabSparseMatrix.s_arrayNzmax*4%8 != 0)
    out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_rowIndexPadding), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_columnIndexTag), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_columnIndexLength), 4);
  for (unsigned int i=0; i<m_MatlabSparseMatrix.s_dimensionNbColumn+1; ++i)
    out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_columnIndex[i]), 4);
  if ((m_MatlabSparseMatrix.s_dimensionNbColumn+1)*4%8 != 0)
    out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_columnIndexPadding), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_valueTag), 4);
  out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_valueLength), 4);
  for (unsigned int i=0; i<m_MatlabSparseMatrix.s_arrayNzmax; ++i)
    out.write(reinterpret_cast<const char*>(&m_MatlabSparseMatrix.s_value[i]), 8);
}

void rtk::MatlabSparseMatrix::Print() {
  std::cout << "m_MatlabSparseMatrix.s_headerMatlab : \"" << m_MatlabSparseMatrix.s_headerMatlab << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_headerOffset : \"" << m_MatlabSparseMatrix.s_headerOffset << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_headerVersion : \"" << std::hex << +m_MatlabSparseMatrix.s_headerVersion << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_headerEndian : \"" << m_MatlabSparseMatrix.s_headerEndian << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_mainTag : \"" << m_MatlabSparseMatrix.s_mainTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_dataLength : \"" << m_MatlabSparseMatrix.s_dataLength << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_arrayTag : \"" << m_MatlabSparseMatrix.s_arrayTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_arrayLength : \"" << m_MatlabSparseMatrix.s_arrayLength << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_arrayFlags : \"" << (unsigned int)m_MatlabSparseMatrix.s_arrayFlags << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_arrayClass : \"" << (unsigned int)m_MatlabSparseMatrix.s_arrayClass << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_arrayUndefined : \"" << m_MatlabSparseMatrix.s_arrayUndefined << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_arrayNzmax : \"" << m_MatlabSparseMatrix.s_arrayNzmax << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_dimensionTag : \"" << m_MatlabSparseMatrix.s_dimensionTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_dimensionLength : \"" << m_MatlabSparseMatrix.s_dimensionLength << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_dimensionNbRow : \"" << m_MatlabSparseMatrix.s_dimensionNbRow << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_dimensionNbColumn : \"" << m_MatlabSparseMatrix.s_dimensionNbColumn << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_nameLength : \"" << m_MatlabSparseMatrix.s_nameLength << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_nameTag : \"" << m_MatlabSparseMatrix.s_nameTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_nameChar : \"" << m_MatlabSparseMatrix.s_nameChar << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_namePadding : \"" << m_MatlabSparseMatrix.s_namePadding << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_rowIndexTag : \"" << m_MatlabSparseMatrix.s_rowIndexTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_rowIndexLength : \"" << m_MatlabSparseMatrix.s_rowIndexLength << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_rowIndex : \"";
  for (unsigned int i=0; i<m_MatlabSparseMatrix.s_arrayNzmax; ++i)
    std::cout << m_MatlabSparseMatrix.s_rowIndex[i] << " ";
  std::cout << "\"" << std::endl;
  if (m_MatlabSparseMatrix.s_arrayNzmax*4%8 != 0)
    std::cout << "m_MatlabSparseMatrix.s_rowIndexPadding : \"" << m_MatlabSparseMatrix.s_rowIndexPadding << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_columnIndexTag : \"" << m_MatlabSparseMatrix.s_columnIndexTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_columnIndexLength : \"" << m_MatlabSparseMatrix.s_columnIndexLength << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_columnIndex : \"";
  for (unsigned int i=0; i<m_MatlabSparseMatrix.s_dimensionNbColumn+1; ++i)
    std::cout << m_MatlabSparseMatrix.s_columnIndex[i] << " ";
  std::cout << "\"" << std::endl;
  if ((m_MatlabSparseMatrix.s_dimensionNbColumn+1)*4%8 != 0)
    std::cout << "m_MatlabSparseMatrix.s_columnIndexPadding : \"" << m_MatlabSparseMatrix.s_columnIndexPadding << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_valueTag : \"" << m_MatlabSparseMatrix.s_valueTag << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_valueLength : \"" << m_MatlabSparseMatrix.s_valueLength << "\"" << std::endl;
  std::cout << "m_MatlabSparseMatrix.s_value : \"";
  for (unsigned int i=0; i<m_MatlabSparseMatrix.s_arrayNzmax; ++i)
    std::cout << m_MatlabSparseMatrix.s_value[i] << " ";
  std::cout << "\"" << std::endl;
}

/*
std::istream& operator>>(std::istream& in) {
  char temp1(0);
  unsigned short int temp2(0);
  unsigned long int temp4(0);
  double temp8(0);

  //header
  for (unsigned int i=0; i<116; ++i) {
    in.read(reinterpret_cast<char*>(&temp1),1);
    m_MatlabSparseMatrix.s_headerMatlab[i] = temp1;
  }
  std::cout << "m_MatlabSparseMatrix.s_headerMatlab : \"" << m_MatlabSparseMatrix.s_headerMatlab << "\"" << std::endl;
  for (unsigned int i=0; i<8; ++i) {
    in.read(reinterpret_cast<char*>(&temp1),1);
    m_MatlabSparseMatrix.s_headerOffset[i] = temp1;
  }
  std::cout << "m_MatlabSparseMatrix.s_headerOffset : \"" << m_MatlabSparseMatrix.s_headerOffset << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp2), 2);
  m_MatlabSparseMatrix.s_headerVersion = temp2;
  std::cout << "m_MatlabSparseMatrix.s_headerVersion : \"" << std::hex << +m_MatlabSparseMatrix.s_headerVersion << "\"" << std::endl;
  for (unsigned int i=0; i<2; ++i) {
    in.read(reinterpret_cast<char*>(&temp1),1);
    m_MatlabSparseMatrix.s_headerEndian[i] = temp1;
  }
  std::cout << "m_MatlabSparseMatrix.s_headerEndian : \"" << m_MatlabSparseMatrix.s_headerEndian << "\"" << std::endl;
  //data
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_mainTag = temp4;
  std::cout << "m_MatlabSparseMatrix.s_mainTag : \"" << m_MatlabSparseMatrix.s_mainTag << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_dataLength = temp4;
  std::cout << "m_MatlabSparseMatrix.s_dataLength : \"" << m_MatlabSparseMatrix.s_dataLength << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_arrayTag = temp4;
  std::cout << "m_MatlabSparseMatrix.s_arrayTag : \"" << m_MatlabSparseMatrix.s_arrayTag << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_arrayLength = temp4;
  std::cout << "m_MatlabSparseMatrix.s_arrayLength : \"" << m_MatlabSparseMatrix.s_arrayLength << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp1), 1);
  m_MatlabSparseMatrix.s_arrayFlags = temp1;
  std::cout << "m_MatlabSparseMatrix.s_arrayFlags : \"" << (unsigned int)m_MatlabSparseMatrix.s_arrayFlags << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp1),1);
  m_MatlabSparseMatrix.s_arrayClass = temp1;
  std::cout << "m_MatlabSparseMatrix.s_arrayClass : \"" << (unsigned int)m_MatlabSparseMatrix.s_arrayClass << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp2), 2);
  m_MatlabSparseMatrix.s_arrayUndefined = temp2;
  std::cout << "m_MatlabSparseMatrix.s_arrayUndefined : \"" << m_MatlabSparseMatrix.s_arrayUndefined << "\"" << std::endl;
in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_arrayNzmax = temp4;
  std::cout << "m_MatlabSparseMatrix.s_arrayNzmax : \"" << m_MatlabSparseMatrix.s_arrayNzmax << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_dimensionTag = temp4;
  std::cout << "m_MatlabSparseMatrix.s_dimensionTag : \"" << m_MatlabSparseMatrix.s_dimensionTag << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_dimensionLength = temp4;
  std::cout << "m_MatlabSparseMatrix.s_dimensionLength : \"" << m_MatlabSparseMatrix.s_dimensionLength << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_dimensionNbRow = temp4;
  std::cout << "m_MatlabSparseMatrix.s_dimensionNbRow : \"" << m_MatlabSparseMatrix.s_dimensionNbRow << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_dimensionNbColumn = temp4;
  std::cout << "m_MatlabSparseMatrix.s_dimensionNbColumn : \"" << m_MatlabSparseMatrix.s_dimensionNbColumn << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp2), 2);
  m_MatlabSparseMatrix.s_nameLength = temp2;
  std::cout << "m_MatlabSparseMatrix.s_nameLength : \"" << m_MatlabSparseMatrix.s_nameLength << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp2), 2);
  m_MatlabSparseMatrix.s_nameTag = temp2;
  std::cout << "m_MatlabSparseMatrix.s_nameTag : \"" << m_MatlabSparseMatrix.s_nameTag << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp1),1);
  m_MatlabSparseMatrix.s_nameChar = temp1;
  std::cout << "m_MatlabSparseMatrix.s_nameChar : \"" << m_MatlabSparseMatrix.s_nameChar << "\"" << std::endl;
  for (unsigned int i=0; i<3; ++i) {
    in.read(reinterpret_cast<char*>(&temp1),1);
    m_MatlabSparseMatrix.s_namePadding[i] = temp1;
  }
  std::cout << "m_MatlabSparseMatrix.s_namePadding : \"" << m_MatlabSparseMatrix.s_namePadding << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_rowIndexTag = temp4;
  std::cout << "m_MatlabSparseMatrix.s_rowIndexTag : \"" << m_MatlabSparseMatrix.s_rowIndexTag << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_rowIndexLength = temp4;
  std::cout << "m_MatlabSparseMatrix.s_rowIndexLength : \"" << m_MatlabSparseMatrix.s_rowIndexLength << "\"" << std::endl;
  m_MatlabSparseMatrix.s_rowIndex = new unsigned long int[m_MatlabSparseMatrix.s_arrayNzmax];
  std::cout << "m_MatlabSparseMatrix.s_rowIndex : \"";
  for (unsigned int i=0; i<m_MatlabSparseMatrix.s_arrayNzmax; ++i) {
    in.read(reinterpret_cast<char*>(&temp4),4);
    m_MatlabSparseMatrix.s_rowIndex[i] = temp4;
    std::cout << m_MatlabSparseMatrix.s_rowIndex[i] << " ";
  }
  std::cout << "\"" << std::endl;
  if (m_MatlabSparseMatrix.s_arrayNzmax*4%8 != 0) {
    in.read(reinterpret_cast<char*>(&temp4), 4);
    m_MatlabSparseMatrix.s_rowIndexPadding = temp4;
    std::cout << "m_MatlabSparseMatrix.s_rowIndexPadding : \"" << m_MatlabSparseMatrix.s_rowIndexPadding << "\"" << std::endl;
  }
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_columnIndexTag = temp4;
  std::cout << "m_MatlabSparseMatrix.s_columnIndexTag : \"" << m_MatlabSparseMatrix.s_columnIndexTag << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_columnIndexLength = temp4;
  std::cout << "m_MatlabSparseMatrix.s_columnIndexLength : \"" << m_MatlabSparseMatrix.s_columnIndexLength << "\"" << std::endl;
  m_MatlabSparseMatrix.s_columnIndex = new unsigned long int[m_MatlabSparseMatrix.s_dimensionNbColumn+1];
  std::cout << "m_MatlabSparseMatrix.s_columnIndex : \"";
  for (unsigned int i=0; i<m_MatlabSparseMatrix.s_dimensionNbColumn+1; ++i) {
    in.read(reinterpret_cast<char*>(&temp4),4);
    m_MatlabSparseMatrix.s_columnIndex[i] = temp4;
    std::cout << m_MatlabSparseMatrix.s_columnIndex[i] << " ";
  }
  std::cout << "\"" << std::endl;
  if ((m_MatlabSparseMatrix.s_dimensionNbColumn+1)*4%8 != 0) {
    in.read(reinterpret_cast<char*>(&temp4), 4);
    m_MatlabSparseMatrix.s_columnIndexPadding = temp4;
    std::cout << "m_MatlabSparseMatrix.s_columnIndexPadding : \"" << m_MatlabSparseMatrix.s_columnIndexPadding << "\"" << std::endl;
  }
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_valueTag = temp4;
  std::cout << "m_MatlabSparseMatrix.s_valueTag : \"" << m_MatlabSparseMatrix.s_valueTag << "\"" << std::endl;
  in.read(reinterpret_cast<char*>(&temp4), 4);
  m_MatlabSparseMatrix.s_valueLength = temp4;
  std::cout << "m_MatlabSparseMatrix.s_valueLength : \"" << m_MatlabSparseMatrix.s_valueLength << "\"" << std::endl;
  m_MatlabSparseMatrix.s_value = new double[m_MatlabSparseMatrix.s_arrayNzmax];
  std::cout << "m_MatlabSparseMatrix.s_value : \"";
  for (unsigned int i=0; i<m_MatlabSparseMatrix.s_arrayNzmax; ++i) {
    in.read(reinterpret_cast<char*>(&temp8),8);
    m_MatlabSparseMatrix.s_value[i] = temp8;
    std::cout << m_MatlabSparseMatrix.s_value[i] << " ";
  }
  std::cout << "\"" << std::endl;

  return in;
}
*/