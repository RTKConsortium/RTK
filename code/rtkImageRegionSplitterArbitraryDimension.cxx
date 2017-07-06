/*=========================================================================
 *
 *  Copyright Insight Software Consortium
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

#include "rtkImageRegionSplitterArbitraryDimension.h"


namespace rtk
{

ImageRegionSplitterArbitraryDimension
::ImageRegionSplitterArbitraryDimension()
{
  m_SplitAxis = 0;
}

unsigned int
ImageRegionSplitterArbitraryDimension
::GetNumberOfSplitsInternal(unsigned int itkNotUsed(dim),
                            const itk::IndexValueType itkNotUsed(regionIndex)[],
                            const itk::SizeValueType regionSize[],
                            unsigned int requestedNumber) const
{
  // determine the actual number of pieces that will be generated
  const double range = regionSize[m_SplitAxis];
  const unsigned int valuesPerPiece = itk::Math::Ceil< unsigned int >(range / (double)requestedNumber);
  const unsigned int maxPieceUsed = itk::Math::Ceil< unsigned int >(range / (double)valuesPerPiece) - 1;

  return maxPieceUsed + 1;
}

unsigned int
ImageRegionSplitterArbitraryDimension
::GetSplitInternal(unsigned int itkNotUsed(dim),
                   unsigned int i,
                   unsigned int numberOfPieces,
                   itk::IndexValueType regionIndex[],
                   itk::SizeValueType regionSize[]) const
{
  // determine the actual number of pieces that will be generated
  const double range=static_cast<double>(regionSize[m_SplitAxis]);
  const unsigned int valuesPerPiece = itk::Math::Ceil< unsigned int >(range / static_cast<double>(numberOfPieces));
  const unsigned int maxPieceIdUsed = itk::Math::Ceil< unsigned int >(range / static_cast<double>(valuesPerPiece)) - 1;

  // Split the region
  if ( i < maxPieceIdUsed )
    {
    regionIndex[m_SplitAxis] += i * valuesPerPiece;
    regionSize[m_SplitAxis] = valuesPerPiece;
    }
  if ( i == maxPieceIdUsed )
    {
    regionIndex[m_SplitAxis] += i * valuesPerPiece;
    // last piece needs to process the "rest" dimension being split
    regionSize[m_SplitAxis] = regionSize[m_SplitAxis] - i * valuesPerPiece;
    }

  return maxPieceIdUsed + 1;

}

}
