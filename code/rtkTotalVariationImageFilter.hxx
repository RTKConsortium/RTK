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

#ifndef rtkTotalVariationImageFilter_hxx
#define rtkTotalVariationImageFilter_hxx
#include "rtkTotalVariationImageFilter.h"


#include "itkConstNeighborhoodIterator.h"
#include "itkProgressReporter.h"


namespace rtk
{

template< typename TInputImage >
TotalVariationImageFilter< TInputImage >
::TotalVariationImageFilter()
{
  // first output is a copy of the image, DataObject created by
  // superclass
  m_SumOfSquareRoots.Fill(1);

  // allocate the data object for the output which is
  // just a decorator around real type
    typename RealObjectType::Pointer output =
      static_cast< RealObjectType * >( this->MakeOutput(1).GetPointer() );
    this->itk::ProcessObject::SetNthOutput( 1, output.GetPointer() );

  this->GetTotalVariationOutput()->Set(itk::NumericTraits< RealType >::Zero);

  // Initialize
  m_UseImageSpacing = true;
}

template< typename TInputImage >
itk::DataObject::Pointer
TotalVariationImageFilter< TInputImage >
::MakeOutput(DataObjectPointerArraySizeType output)
{
  switch ( output )
    {
    case 0:
      return TInputImage::New().GetPointer();
      break;
    case 1:
      return RealObjectType::New().GetPointer();
      break;
    default:
      // might as well make an image
      return TInputImage::New().GetPointer();
      break;
    }
}

template< typename TInputImage >
typename TotalVariationImageFilter< TInputImage >::RealObjectType *
TotalVariationImageFilter< TInputImage >
::GetTotalVariationOutput()
{
  return static_cast< RealObjectType * >( this->itk::ProcessObject::GetOutput(1) );
}

template< typename TInputImage >
const typename TotalVariationImageFilter< TInputImage >::RealObjectType *
TotalVariationImageFilter< TInputImage >
::GetTotalVariationOutput() const
{
  return static_cast< const RealObjectType * >( this->itk::ProcessObject::GetOutput(1) );
}

template< typename TInputImage >
void
TotalVariationImageFilter< TInputImage >
::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();
  if ( this->GetInput() )
    {
    InputImagePointer image =
      const_cast< typename Superclass::InputImageType * >( this->GetInput() );
    image->SetRequestedRegionToLargestPossibleRegion();
    }
}

template< typename TInputImage >
void
TotalVariationImageFilter< TInputImage >
::EnlargeOutputRequestedRegion(itk::DataObject *data)
{
  Superclass::EnlargeOutputRequestedRegion(data);
  data->SetRequestedRegionToLargestPossibleRegion();
}

template< typename TInputImage >
void
TotalVariationImageFilter< TInputImage >
::AllocateOutputs()
{
  // Pass the input through as the output
  InputImagePointer image = const_cast< TInputImage * >( this->GetInput() );

  this->GraftOutput(image);

  // Nothing that needs to be allocated for the remaining outputs
}

template< typename TInputImage >
void
TotalVariationImageFilter< TInputImage >
::BeforeThreadedGenerateData()
{
  itk::ThreadIdType numberOfThreads = this->GetNumberOfThreads();

  // Resize the thread temporaries
  m_SumOfSquareRoots.SetSize(numberOfThreads);

  // Initialize the temporaries
  m_SumOfSquareRoots.Fill(itk::NumericTraits< RealType >::Zero);
}

template< typename TInputImage >
void
TotalVariationImageFilter< TInputImage >
::AfterThreadedGenerateData()
{
  RealType totalVariation = 0;
  itk::ThreadIdType numberOfThreads = this->GetNumberOfThreads();

  // Add up the results from all threads
  for (itk::ThreadIdType i = 0; i < numberOfThreads; i++ )
    {
    totalVariation += m_SumOfSquareRoots[i];
    }

  // Set the output
  this->GetTotalVariationOutput()->Set(totalVariation);
}

template< typename TInputImage >
void
TotalVariationImageFilter< TInputImage >
::ThreadedGenerateData(const RegionType & outputRegionForThread,
                       itk::ThreadIdType threadId)
{
  const itk::SizeValueType size0 = outputRegionForThread.GetSize(0);
  if( size0 == 0)
    {
    return;
    }
  RealType sumOfSquareRoots = itk::NumericTraits< RealType >::Zero;
  typename TInputImage::ConstPointer input = this->GetInput(0);

  itk::Size<ImageDimension> radius;
  radius.Fill(1);

  itk::ConstNeighborhoodIterator<TInputImage> iit(radius, input, outputRegionForThread);
  iit.GoToBegin();
  itk::ZeroFluxNeumannBoundaryCondition<TInputImage> boundaryCondition;
  iit.OverrideBoundaryCondition(&boundaryCondition);

  itk::SizeValueType c = (itk::SizeValueType) (iit.Size() / 2); // get offset of center pixel
  itk::SizeValueType strides[ImageDimension]; // get offsets to access neighboring pixels
  itk::Vector<RealType, ImageDimension> invSpacingCoeffs;
  for (unsigned int dim=0; dim<ImageDimension; dim++)
    {
    strides[dim] = iit.GetStride(dim);

    invSpacingCoeffs[dim] = 1;
    if ( m_UseImageSpacing == true )
      {
      invSpacingCoeffs[dim] = 1.0 / this->GetOutput()->GetSpacing()[dim];
      }
    }

  // Run through the image
  while(!iit.IsAtEnd())
    {
    // Compute the local differences around the central pixel
    float difference;
    float sumOfSquaredDifferences = 0;
    for (unsigned int dim = 0; dim < ImageDimension; dim++)
      {
      difference = (iit.GetPixel(c + strides[dim]) - iit.GetPixel(c)) * invSpacingCoeffs[dim];
      sumOfSquaredDifferences += difference * difference;
      }
    sumOfSquareRoots += sqrt(sumOfSquaredDifferences);
    ++iit;
    }

  m_SumOfSquareRoots[threadId] = sumOfSquareRoots;
}

template< typename TInputImage >
void
TotalVariationImageFilter< TInputImage >
::PrintSelf(std::ostream & os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Total Variation: " << this->GetTotalVariation() << std::endl;
}
} // end namespace rtk
#endif
