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

#ifndef rtkRayQuadricIntersectionImageFilter_hxx
#define rtkRayQuadricIntersectionImageFilter_hxx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include "rtkHomogeneousMatrix.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
RayQuadricIntersectionImageFilter<TInputImage,TOutputImage>
::RayQuadricIntersectionImageFilter():
  m_RQIFunctor( RQIFunctionType::New() ),
  m_Geometry(ITK_NULLPTR),
  m_Density(1.)
{
}

template <class TInputImage, class TOutputImage>
RayQuadricIntersectionImageFilter<TInputImage,TOutputImage>::RQIFunctionType::Pointer
RayQuadricIntersectionImageFilter<TInputImage,TOutputImage>
::GetRQIFunctor()
{
  this->Modified();
  return this->m_RQIFunctor;
}

template <class TInputImage, class TOutputImage>
void
RayQuadricIntersectionImageFilter<TInputImage,TOutputImage>
::BeforeThreadedGenerateData()
{
  if(this->GetGeometry()->GetGantryAngles().size() !=
          this->GetOutput()->GetLargestPossibleRegion().GetSize()[2])
      itkExceptionMacro(<<"Number of projections in the input stack and the geometry object differ.")
}

template <class TInputImage, class TOutputImage>
void
RayQuadricIntersectionImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType itkNotUsed(threadId) )
{
  // Create local object for multithreading purposes
  RQIFunctionType::Pointer rqiFunctor = RQIFunctionType::New();
  rqiFunctor->SetA( m_RQIFunctor->GetA() );
  rqiFunctor->SetB( m_RQIFunctor->GetB() );
  rqiFunctor->SetC( m_RQIFunctor->GetC() );
  rqiFunctor->SetD( m_RQIFunctor->GetD() );
  rqiFunctor->SetE( m_RQIFunctor->GetE() );
  rqiFunctor->SetF( m_RQIFunctor->GetF() );
  rqiFunctor->SetG( m_RQIFunctor->GetG() );
  rqiFunctor->SetH( m_RQIFunctor->GetH() );
  rqiFunctor->SetI( m_RQIFunctor->GetI() );
  rqiFunctor->SetJ( m_RQIFunctor->GetJ() );

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

    rqiFunctor->SetRayOrigin( itIn->GetSourcePosition() );

    // Compute ray intersection length
    if( rqiFunctor->Evaluate(itIn->GetDirection()) )
      itOut.Set( itIn->Get() + m_Density*(rqiFunctor->GetFarthestDistance() - rqiFunctor->GetNearestDistance() ));
    else
      itOut.Set( itIn->Get() );
    }

  delete itIn;
}

} // end namespace rtk

#endif
