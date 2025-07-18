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

#ifndef rtkForwardDifferenceGradientImageFilter_hxx
#define rtkForwardDifferenceGradientImageFilter_hxx

#include <itkConstNeighborhoodIterator.h>
#include <itkNeighborhoodInnerProduct.h>
#include <itkImageRegionIterator.h>
#include <itkForwardDifferenceOperator.h>
#include <itkNeighborhoodAlgorithm.h>
#include <itkOffset.h>
#include <itkProgressReporter.h>

namespace rtk
{
//
// Constructor
//
template <typename TInputImage, typename TOperatorValueType, typename TOuputValue, typename TOuputImage>
ForwardDifferenceGradientImageFilter<TInputImage, TOperatorValueType, TOuputValue, TOuputImage>::
  ForwardDifferenceGradientImageFilter()
{
  // default boundary condition
  m_BoundaryCondition = new itk::ZeroFluxNeumannBoundaryCondition<TInputImage>();
  m_IsBoundaryConditionOverriden = false;

  // default behaviour is to take into account both spacing and direction
  this->m_UseImageSpacing = true;
  this->m_UseImageDirection = true;

  // default behaviour is to process all dimensions
  for (unsigned int dim = 0; dim < TInputImage::ImageDimension; dim++)
  {
    m_DimensionsProcessed[dim] = true;
  }
}

//
// Destructor
//
template <typename TInputImage, typename TOperatorValueType, typename TOuputValue, typename TOuputImage>
ForwardDifferenceGradientImageFilter<TInputImage, TOperatorValueType, TOuputValue, TOuputImage>::
  ~ForwardDifferenceGradientImageFilter()
{
  delete m_BoundaryCondition;
}

// This should be handled by an itkMacro, but it doesn't seem to work with pointer types
template <typename TInputImage, typename TOperatorValueType, typename TOuputValue, typename TOuputImage>
void
ForwardDifferenceGradientImageFilter<TInputImage, TOperatorValueType, TOuputValue, TOuputImage>::SetDimensionsProcessed(
  bool * DimensionsProcessed)
{
  bool bModif = false;
  for (unsigned int dim = 0; dim < TInputImage::ImageDimension; dim++)
  {
    if (m_DimensionsProcessed[dim] != DimensionsProcessed[dim])
    {
      m_DimensionsProcessed[dim] = DimensionsProcessed[dim];
      bModif = true;
    }
  }
  if (bModif)
    this->Modified();
}

template <typename TInputImage, typename TOperatorValueType, typename TOuputValue, typename TOuputImage>
void
ForwardDifferenceGradientImageFilter<TInputImage, TOperatorValueType, TOuputValue, TOuputImage>::
  OverrideBoundaryCondition(itk::ImageBoundaryCondition<TInputImage> * boundaryCondition)
{
  delete m_BoundaryCondition;
  m_BoundaryCondition = boundaryCondition;
  m_IsBoundaryConditionOverriden = true;
}

template <typename TInputImage, typename TOperatorValueType, typename TOuputValue, typename TOuputImage>
void
ForwardDifferenceGradientImageFilter<TInputImage, TOperatorValueType, TOuputValue, TOuputImage>::
  GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // get pointers to the input and output
  InputImagePointer  inputPtr = const_cast<InputImageType *>(this->GetInput());
  OutputImagePointer outputPtr = this->GetOutput();

  if (!inputPtr || !outputPtr)
  {
    return;
  }

  // Build an operator so that we can determine the kernel size
  itk::ForwardDifferenceOperator<OperatorValueType, InputImageDimension> oper;
  oper.SetDirection(0);
  oper.CreateDirectional();
  const itk::SizeValueType radius = oper.GetRadius()[0];

  // get a copy of the input requested region (should equal the output
  // requested region)
  typename TInputImage::RegionType inputRequestedRegion = inputPtr->GetRequestedRegion();

  // pad the input requested region by the operator radius
  inputRequestedRegion.PadByRadius(radius);

  // crop the input requested region at the input's largest possible region
  if (inputRequestedRegion.Crop(inputPtr->GetLargestPossibleRegion()))
  {
    inputPtr->SetRequestedRegion(inputRequestedRegion);
    return;
  }
  else
  {
    // Couldn't crop the region (requested region is outside the largest
    // possible region).  Throw an exception.

    // store what we tried to request (prior to trying to crop)
    inputPtr->SetRequestedRegion(inputRequestedRegion);

    // build an exception
    itk::InvalidRequestedRegionError e(__FILE__, __LINE__);
    e.SetLocation(ITK_LOCATION);
    e.SetDescription("Requested region is (at least partially) outside the largest possible region.");
    e.SetDataObject(inputPtr);
    throw e;
  }
}

template <typename TInputImage, typename TOperatorValueType, typename TOuputValue, typename TOuputImage>
void
ForwardDifferenceGradientImageFilter<TInputImage, TOperatorValueType, TOuputValue, TOuputImage>::
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread)
{

  itk::NeighborhoodInnerProduct<InputImageType, OperatorValueType, OutputValueType> SIP;

  // Get the input and output
  OutputImageType *      outputImage = this->GetOutput();
  const InputImageType * inputImage = this->GetInput();

  // Generate a list of indices of the dimensions to process
  std::vector<int> dimsToProcess;
  //  std::vector<int> dimsNotToProcess;
  for (unsigned int dim = 0; dim < TInputImage::ImageDimension; dim++)
  {
    if (m_DimensionsProcessed[dim])
      dimsToProcess.push_back(dim);
    //    else dimsNotToProcess.push_back(dim);
  }

  // Set up operators
  itk::ForwardDifferenceOperator<OperatorValueType, InputImageDimension> op[InputImageDimension];

  for (unsigned int i = 0; i < InputImageDimension; i++)
  {
    op[i].SetDirection(0);
    op[i].CreateDirectional();

    // Take into account the pixel spacing if necessary
    if (m_UseImageSpacing == true)
    {
      if (this->GetInput()->GetSpacing()[i] == 0.0)
      {
        itkExceptionMacro(<< "Image spacing cannot be zero.");
      }
      else
      {
        op[i].ScaleCoefficients(1.0 / this->GetInput()->GetSpacing()[i]);
      }
    }
  }

  // Calculate iterator radius
  itk::Size<InputImageDimension> radius;
  for (unsigned int i = 0; i < InputImageDimension; ++i)
  {
    radius[i] = op[0].GetRadius()[0];
  }

  // Find the data-set boundary "faces"
  itk::NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>                        bC;
  typename itk::NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>::FaceListType faceList =
    bC(inputImage, outputRegionForThread, radius);

  auto fit = faceList.begin();

  // Initialize the x_slice array
  auto nit = itk::ConstNeighborhoodIterator<InputImageType>(radius, inputImage, *fit);

  std::slice               x_slice[InputImageDimension];
  const itk::SizeValueType center = nit.Size() / 2;
  for (unsigned int i = 0; i < InputImageDimension; ++i)
  {
    x_slice[i] = std::slice(center - nit.GetStride(i) * radius[i], op[i].GetSize()[0], nit.GetStride(i));
  }

  CovariantVectorType gradient(itk::NumericTraits<typename CovariantVectorType::ValueType>::ZeroValue());
  // Process non-boundary face and then each of the boundary faces.
  // These are N-d regions which border the edge of the buffer.
  for (auto & currentFace : faceList)
  {
    nit = itk::ConstNeighborhoodIterator<InputImageType>(radius, inputImage, currentFace);
    auto it = itk::ImageRegionIterator<OutputImageType>(outputImage, currentFace);
    nit.OverrideBoundaryCondition(m_BoundaryCondition);
    nit.GoToBegin();

    while (!nit.IsAtEnd())
    {
      for (std::vector<int>::size_type i = 0; i < dimsToProcess.size(); ++i)
      {
        gradient[i] = SIP(x_slice[dimsToProcess[i]], nit, op[dimsToProcess[i]]);
      }
      //      for ( i = 0; i < dimsNotToProcess.size(); ++i )
      //        {
      //        gradient[dimsNotToProcess[i]] = 0;
      //        }

      // This method optionally performs a tansform for Physical
      // coordinates and potential conversion to a different output
      // pixel type.
      this->SetOutputPixel(it, gradient);

      ++nit;
      ++it;
    }
  }
}

template <typename TInputImage, typename TOperatorValueType, typename TOuputValue, typename TOuputImage>
void
ForwardDifferenceGradientImageFilter<TInputImage, TOperatorValueType, TOuputValue, TOuputImage>::
  GenerateOutputInformation()
{
  // this methods is overloaded so that if the output image is a
  // VectorImage then the correct number of components are set.

  Superclass::GenerateOutputInformation();
  OutputImageType * output = this->GetOutput();

  if (!output)
  {
    return;
  }
  if (output->GetNumberOfComponentsPerPixel() != InputImageDimension)
  {
    output->SetNumberOfComponentsPerPixel(InputImageDimension);
  }
}


/**
 * Standard "PrintSelf" method
 */
template <typename TInputImage, typename TOperatorValueType, typename TOuputValue, typename TOuputImage>
void
ForwardDifferenceGradientImageFilter<TInputImage, TOperatorValueType, TOuputValue, TOuputImage>::PrintSelf(
  std::ostream & os,
  itk::Indent    indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "UseImageSpacing: " << (this->m_UseImageSpacing ? "On" : "Off") << std::endl;
  os << indent << "UseImageDirection = " << (this->m_UseImageDirection ? "On" : "Off") << std::endl;
}
} // namespace rtk

#endif
