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

#ifndef rtkGeneralPurposeFunctions_h
#define rtkGeneralPurposeFunctions_h

#include <vector>
#include <itkMacro.h>
#include <itkImageFileWriter.h>
#include <itkMath.h>

namespace rtk
{

/** \brief Sorts a vector and returns the sorting permutation
 *
 * This function takes a vector and returns another vector
 * containing the permutated indices, not the sorted values.
 *
 * \author Cyril Mory
 *
 * \ingroup Functions
 */
template< typename TVectorElement >
std::vector<unsigned int>
GetSortingPermutation(std::vector<TVectorElement> input)
{
  // Define a vector of pairs (value and index)
  std::vector<std::pair<TVectorElement, unsigned int> > pairsVector;

  // Fill it
  for (unsigned int i = 0; i < input.size(); i++)
      pairsVector.push_back(std::make_pair(input[i], i));

  // Sort it according to values
  std::sort(pairsVector.begin(), pairsVector.end());

  // Extract the permutated indices
  std::vector<unsigned int> result;
  for (unsigned int i = 0; i < pairsVector.size(); i++)
      result.push_back(pairsVector[i].second);

  // Return
  return result;
}

std::vector<double>
ReadSignalFile(std::string filename)
{
  std::vector<double> signalVector;
  std::ifstream is( filename.c_str() );
  if( !is.is_open() )
    {
    itkGenericExceptionMacro(<< "Could not open signal file " << filename);
    }

  double value;
  std::string s;
  while (getline(is, s))
    {
    if (!s.empty())
      {
      std::istringstream tmp(s);
      tmp >> value;
      signalVector.push_back(itk::Math::Round<double>(value * 100) / 100);
      }
    }

  return signalVector;
}

template< typename ImageType >
void
WriteImage(typename ImageType::Pointer input, std::string name)
{
  //Create an itk::ImageFileWriter
  typedef itk::ImageFileWriter<ImageType> WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetInput(input);
  writer->SetFileName(name);
  writer->Update();
}

}

#endif // rtkGeneralPurposeFunctions_h
