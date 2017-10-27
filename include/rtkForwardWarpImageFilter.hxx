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

#ifndef rtkForwardWarpImageFilter_hxx
#define rtkForwardWarpImageFilter_hxx

#include "rtkForwardWarpImageFilter.h"
#include "rtkHomogeneousMatrix.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkNeighborhoodIterator.h>
#include <itkConstantBoundaryCondition.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMacro.h>

namespace rtk
{

template <class TInputImage, class TOutputImage, class TDVF>
ForwardWarpImageFilter<TInputImage, TOutputImage, TDVF>
::ForwardWarpImageFilter()
{
  m_Protected_DefFieldSizeSame = false;
  m_Protected_EndIndex.Fill(0);
  m_Protected_StartIndex.Fill(0);
}

template <class TInputImage, class TOutputImage, class TDVF>
void
ForwardWarpImageFilter<TInputImage, TOutputImage, TDVF>
::Protected_EvaluateDisplacementAtPhysicalPoint(const PointType & point, DisplacementType &output)
{

  DisplacementFieldPointer fieldPtr = this->GetDisplacementField();

  itk::ContinuousIndex< double, TDVF::ImageDimension > index;
  fieldPtr->TransformPhysicalPointToContinuousIndex(point, index);
  unsigned int dim;  // index over dimension
  /**
   * Compute base index = closest index below point
   * Compute distance from point to base index
   */
  typename TDVF::IndexType baseIndex;
  typename TDVF::IndexType neighIndex;
  double    distance[TDVF::ImageDimension];

  for ( dim = 0; dim < TDVF::ImageDimension; dim++ )
    {
    baseIndex[dim] = itk::Math::Floor< typename TDVF::IndexValueType >(index[dim]);

    if ( baseIndex[dim] >=  this->m_Protected_StartIndex[dim] )
      {
      if ( baseIndex[dim] <  this->m_Protected_EndIndex[dim] )
        {
        distance[dim] = index[dim] - static_cast< double >( baseIndex[dim] );
        }
      else
        {
        baseIndex[dim] = this->m_Protected_EndIndex[dim];
        distance[dim] = 0.0;
        }
      }
    else
      {
      baseIndex[dim] = this->m_Protected_StartIndex[dim];
      distance[dim] = 0.0;
      }
    }

  /**
   * Interpolated value is the weight some of each of the surrounding
   * neighbors. The weight for each neighbour is the fraction overlap
   * of the neighbor pixel with respect to a pixel centered on point.
   */
  output.Fill(0);

  double       totalOverlap = 0.0;
  unsigned int numNeighbors(1 << TInputImage::ImageDimension);

  for ( unsigned int counter = 0; counter < numNeighbors; counter++ )
    {
    double       overlap = 1.0;    // fraction overlap
    unsigned int upper = counter;  // each bit indicates upper/lower neighbour

    // get neighbor index and overlap fraction
    for ( dim = 0; dim < TDVF::ImageDimension; dim++ )
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

    // get neighbor value only if overlap is not zero
    if ( overlap )
      {
      const DisplacementType input =
        fieldPtr->GetPixel(neighIndex);
      for ( unsigned int k = 0; k < TDVF::ImageDimension; k++ )
        {
        output[k] += overlap * static_cast< double >( input[k] );
        }
      totalOverlap += overlap;
      }

    if ( totalOverlap == 1.0 )
      {
      // finished
      break;
      }
    }
}

template <class TInputImage, class TOutputImage, class TDVF>
void
ForwardWarpImageFilter<TInputImage, TOutputImage, TDVF>
::GenerateData()
{
  Superclass::BeforeThreadedGenerateData();
  DisplacementFieldPointer fieldPtr = this->GetDisplacementField();

  // Connect input image to interpolator
  m_Protected_StartIndex = fieldPtr->GetBufferedRegion().GetIndex();
  for ( unsigned i = 0; i < TDVF::ImageDimension; i++ )
    {
    m_Protected_EndIndex[i] = m_Protected_StartIndex[i]
                    + fieldPtr->GetBufferedRegion().GetSize()[i] - 1;
    }

  typename Superclass::InputImageConstPointer  inputPtr = this->GetInput();
  typename Superclass::OutputImagePointer      outputPtr = this->GetOutput();

  outputPtr->SetRegions(outputPtr->GetRequestedRegion());
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
        inputPtr, inputPtr->GetBufferedRegion());
  itk::ImageRegionIterator< DisplacementFieldType >  fieldIt(fieldPtr, fieldPtr->GetBufferedRegion());
  typename TOutputImage::IndexType        index;
  typename TOutputImage::IndexType        baseIndex;
  typename TOutputImage::IndexType        neighIndex;
  double                                  distance[TInputImage::ImageDimension];
  typename TOutputImage::PointType        point;
  typename Superclass::DisplacementType displacement;
  itk::NumericTraits<typename Superclass::DisplacementType>::SetLength(displacement, TInputImage::ImageDimension);

  unsigned int numNeighbors(1 << TInputImage::ImageDimension);

  // There is a bug in the ITK WarpImageFilter: m_DefFieldSizeSame
  // is computed without taking origin, spacing and direction into
  // account. So we perform a more thorough comparison between
  // output and DVF than in itkWarpImageFilter::BeforeThreadedGenerateData()
  bool skipEvaluateDisplacementAtContinuousIndex =
      ( (outputPtr->GetLargestPossibleRegion() == this->GetDisplacementField()->GetLargestPossibleRegion())
     && (outputPtr->GetSpacing() == this->GetDisplacementField()->GetSpacing())
     && (outputPtr->GetOrigin() == this->GetDisplacementField()->GetOrigin())
     && (outputPtr->GetDirection() == this->GetDisplacementField()->GetDirection())   );

  while ( !inputIt.IsAtEnd() )
    {
    // get the input image index
    index = inputIt.GetIndex();
    inputPtr->TransformIndexToPhysicalPoint(index, point);

    if (skipEvaluateDisplacementAtContinuousIndex)
      displacement = fieldIt.Get();
    else
      this->Protected_EvaluateDisplacementAtPhysicalPoint(point, displacement);

    for ( unsigned int j = 0; j < TInputImage::ImageDimension; j++ )
      point[j] += displacement[j];

    itk::ContinuousIndex< double, TInputImage::ImageDimension > continuousIndexInOutput;
    outputPtr->TransformPhysicalPointToContinuousIndex(point, continuousIndexInOutput);

    // compute the base index in output, ie the closest index below point
    // Check if the baseIndex is in the output's requested region, otherwise skip the splat part
    bool skip = false;

    for ( unsigned int j = 0; j < TInputImage::ImageDimension; j++ )
      {
      baseIndex[j] = itk::Math::Floor<int, double>(continuousIndexInOutput[j]);
      distance[j] = continuousIndexInOutput[j] - static_cast< double >(baseIndex[j]);
      if ( (baseIndex[j] < outputPtr->GetRequestedRegion().GetIndex()[j] - 1) ||
           (baseIndex[j] >= outputPtr->GetRequestedRegion().GetIndex()[j] + (int)outputPtr->GetRequestedRegion().GetSize()[j] ))
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
    ++fieldIt;
    }

  // Divide the output by the accumulated weights, if they are non-zero
  itk::ImageRegionIterator< TOutputImage > outputIt(outputPtr, outputPtr->GetRequestedRegion());
  itk::ImageRegionIterator< TOutputImage > accIt(accumulate, outputPtr->GetRequestedRegion());
  while ( !outputIt.IsAtEnd() )
    {
    if (accIt.Get())
      outputIt.Set(outputIt.Get() / accIt.Get());

    ++outputIt;
    ++accIt;
    }

  // Replace the holes with the weighted mean of their neighbors
  itk::Size<TOutputImage::ImageDimension> radius;
  radius.Fill(3);
  unsigned int pixelsInNeighborhood = 1;
  for (unsigned int dim=0; dim< TOutputImage::ImageDimension; dim++)
    pixelsInNeighborhood *= 2 * radius[dim] + 1;

  itk::NeighborhoodIterator< TOutputImage > outputIt2(radius, outputPtr, outputPtr->GetRequestedRegion());
  itk::NeighborhoodIterator< TOutputImage > accIt2(radius, accumulate, outputPtr->GetRequestedRegion());

  itk::ZeroFluxNeumannBoundaryCondition<TInputImage> zeroFlux;
  outputIt2.OverrideBoundaryCondition(&zeroFlux);

  itk::ConstantBoundaryCondition<TInputImage> constant;
  accIt2.OverrideBoundaryCondition(&constant);

  while ( !outputIt2.IsAtEnd() )
    {
    if (!accIt2.GetCenterPixel())
      {
      // Compute the mean of the neighboring pixels, weighted by the accumulated weights
      typename TOutputImage::PixelType value = 0;
      typename TOutputImage::PixelType weight = 0;
      for (unsigned int idx=0; idx<pixelsInNeighborhood; idx++)
        {
        value += accIt2.GetPixel(idx) * outputIt2.GetPixel(idx);
        weight += accIt2.GetPixel(idx);
        }

      // Replace the hole with this value, or zero (if all surrounding pixels were holes)
      if (weight)
        outputIt2.SetCenterPixel(value / weight);
      else
        outputIt2.SetCenterPixel(0);
      }
    ++outputIt2;
    ++accIt2;
    }

  Superclass::AfterThreadedGenerateData();
}

} // end namespace rtk

#endif //rtkForwardWarpImageFilter_hxx
