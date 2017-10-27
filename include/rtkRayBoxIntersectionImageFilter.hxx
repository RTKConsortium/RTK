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

#ifndef rtkRayBoxIntersectionImageFilter_hxx
#define rtkRayBoxIntersectionImageFilter_hxx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include "rtkHomogeneousMatrix.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage,TOutputImage>
::SetBoxFromImage(OutputImageBaseConstPointer _arg)
{
  m_RBIFunctor->SetBoxFromImage(_arg);
  this->Modified();
}

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage,TOutputImage>
::SetBoxMin(VectorType _boxMin)
{
  m_RBIFunctor->SetBoxMin(_boxMin);
  this->Modified();
}

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage,TOutputImage>
::SetBoxMax(VectorType _boxMax)
{
  m_RBIFunctor->SetBoxMax(_boxMax);
  this->Modified();
}

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType itkNotUsed(threadId) )
{
  // Create local object for multithreading purposes
  RBIFunctionType::Pointer rbiFunctor = RBIFunctionType::New();
  rbiFunctor->SetBoxMin(m_RBIFunctor->GetBoxMin());
  rbiFunctor->SetBoxMax(m_RBIFunctor->GetBoxMax());

  // Iterators on input and output
  typedef ProjectionsRegionConstIteratorRayBased<TInputImage> InputRegionIterator;
  InputRegionIterator *itIn;
  itIn = InputRegionIterator::New(this->GetInput(),
                                  outputRegionForThread,
                                  m_Geometry);
  typedef itk::ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Go over each projection
  for(unsigned int pix=0; pix<outputRegionForThread.GetNumberOfPixels(); pix++, itIn->Next(), ++itOut)
    {
    rbiFunctor->SetRayOrigin( itIn->GetSourcePosition() );

    // Compute ray intersection length
    if( rbiFunctor->Evaluate(itIn->GetDirection()) )
      itOut.Set( itIn->Get() + m_Density*(rbiFunctor->GetFarthestDistance() - rbiFunctor->GetNearestDistance()) );
    else
      itOut.Set( itIn->Get() );
    }

  delete itIn;
}

} // end namespace rtk

#endif
