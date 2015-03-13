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

#ifndef __rtkForwardWarpImageFilter_txx
#define __rtkForwardWarpImageFilter_txx

#include "rtkForwardWarpImageFilter.h"
#include "rtkHomogeneousMatrix.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMacro.h>

namespace rtk
{

template <class TInputImage, class TOutputImage, class TDVF>
ForwardWarpImageFilter<TInputImage, TOutputImage, TDVF>
::ForwardWarpImageFilter()
{
}

template <class TInputImage, class TOutputImage, class TDVF>
void
ForwardWarpImageFilter<TInputImage, TOutputImage, TDVF>
::GenerateData()
{
  Superclass::BeforeThreadedGenerateData();

  typename Superclass::InputImageConstPointer  inputPtr = this->GetInput();
  typename Superclass::OutputImagePointer      outputPtr = this->GetOutput();

  outputPtr->SetRegions(outputPtr->GetLargestPossibleRegion());
  outputPtr->Allocate();
  outputPtr->FillBuffer(0);

  // Allocate an image with the same metadata as the output
  // to accumulate the weights during splat, and divide by the total weights at the end
  typename TOutputImage::Pointer accumulate = TOutputImage::New();
  accumulate->SetRegions(outputPtr->GetRequestedRegion());
  accumulate->Allocate();
  accumulate->FillBuffer(0);

  // iterator for the output image
  itk::ImageRegionConstIteratorWithIndex< TOutputImage > inputIt(
        inputPtr, inputPtr->GetLargestPossibleRegion());
  typename TOutputImage::IndexType        index;
  typename TOutputImage::IndexType        baseIndex;
  typename TOutputImage::IndexType        neighIndex;
  double                                  distance[TInputImage::ImageDimension];
  typename TOutputImage::PointType        point;
  typename Superclass::DisplacementType displacement;
  itk::NumericTraits<typename Superclass::DisplacementType>::SetLength(displacement, TInputImage::ImageDimension);

  unsigned int numNeighbors(1 << TInputImage::ImageDimension);


  // I suspect there is a bug in the ITK code, as m_DefFieldSizeSame
  // is computed without taking origin, spacing and transformation into
  // account. So I commented out the optimized part and used the most generic one

//  if ( this->m_DefFieldSizeSame )
//    {
//    // iterator for the deformation field
//    ImageRegionIterator< DisplacementFieldType >
//    fieldIt(fieldPtr, outputRegionForThread);

//    while ( !outputIt.IsAtEnd() )
//      {
//      // get the output image index
//      index = outputIt.GetIndex();
//      outputPtr->TransformIndexToPhysicalPoint(index, point);

//      // get the required displacement
//      displacement = fieldIt.Get();

//      // compute the required input image point
//      for ( unsigned int j = 0; j < ImageDimension; j++ )
//        {
//        point[j] += displacement[j];
//        }

//      // get the interpolated value
//      if ( m_Interpolator->IsInsideBuffer(point) )
//        {
//        PixelType value =
//          static_cast< PixelType >( m_Interpolator->Evaluate(point) );
//        outputIt.Set(value);
//        }
//      else
//        {
//        outputIt.Set(m_EdgePaddingValue);
//        }
//      ++outputIt;
//      ++fieldIt;
//      progress.CompletedPixel();
//      }
//    }
//  else
//    {
    while ( !inputIt.IsAtEnd() )
      {
      // get the input image index
      index = inputIt.GetIndex();
      inputPtr->TransformIndexToPhysicalPoint(index, point);

      this->EvaluateDisplacementAtPhysicalPoint(point, displacement);

      itk::ContinuousIndex< double, TInputImage::ImageDimension > continuousIndexInInput;
      for ( unsigned int j = 0; j < TInputImage::ImageDimension; j++ )
        point[j] += displacement[j];

      inputPtr->TransformPhysicalPointToContinuousIndex(point, continuousIndexInInput);

      // compute the base index in output, ie the closest index below point
      // Check if the baseIndex is in the output's requested region, otherwise skip the splat part
      bool skip = false;

      for ( unsigned int j = 0; j < TInputImage::ImageDimension; j++ )
        {
        baseIndex[j] = itk::Math::Floor<int, double>(continuousIndexInInput[j]);
        distance[j] = continuousIndexInInput[j] - static_cast< double >(baseIndex[j]);
        if ( (baseIndex[j] < outputPtr->GetRequestedRegion().GetIndex()[j] - 1) ||
             (baseIndex[j] >= outputPtr->GetRequestedRegion().GetIndex()[j] + outputPtr->GetRequestedRegion().GetSize()[j] ))
          skip = true;
        }

      if (!skip)
        {
        // get the splat weights as the overlapping areas between
        for ( unsigned int counter = 0; counter < numNeighbors; counter++ )
          {
          double       overlap = 1.0;    // fraction overlap
          unsigned int upper = counter;  // each bit indicates upper/lower neighbour

          // get neighbor weights as the fraction of overlap
          // of the neighbor pixels with a pixel centered on point
          for ( unsigned int dim = 0; dim < TInputImage::ImageDimension; dim++ )
            {
            if ( upper & 1 )
              {
              neighIndex[dim] = baseIndex[dim] + 1;
              overlap *= distance[dim];
              }
            else
              {
              neighIndex[dim] = baseIndex[dim];
              overlap *= 1.0 - distance[dim];
              }

            upper >>= 1;
            }

          if (outputPtr->GetRequestedRegion().IsInside(neighIndex))
            {
            // Perform splat with this weight, both in output and in the temporary
            // image that accumulates weights
            outputPtr->SetPixel(neighIndex, outputPtr->GetPixel(neighIndex) + overlap * inputIt.Get() );
            accumulate->SetPixel(neighIndex, accumulate->GetPixel(neighIndex) + overlap);
            }
          }
        }

      ++inputIt;
      }

    // Divide the output by the accumulated weights, if they are non-zero
    // Otherwise, zero out the output
    itk::ImageRegionIterator< TOutputImage > outputIt(outputPtr, outputPtr->GetRequestedRegion());
    itk::ImageRegionIterator< TOutputImage > accIt(accumulate, outputPtr->GetRequestedRegion());
    while ( !outputIt.IsAtEnd() )
      {
      if (accIt.Get())
        outputIt.Set(outputIt.Get() / accIt.Get());
      else
        outputIt.Set(0);
      ++outputIt;
      ++accIt;
      }

  Superclass::AfterThreadedGenerateData();
//    }
}

} // end namespace rtk

#endif //__rtkForwardWarpImageFilter_txx
