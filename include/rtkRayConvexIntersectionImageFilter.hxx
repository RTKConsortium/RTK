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

#ifndef rtkRayConvexIntersectionImageFilter_hxx
#define rtkRayConvexIntersectionImageFilter_hxx

#include "math.h"

#include "rtkProjectionsRegionConstIteratorRayBased.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
RayConvexIntersectionImageFilter<TInputImage, TOutputImage>::RayConvexIntersectionImageFilter()
  : m_Geometry(nullptr)
{}

template <class TInputImage, class TOutputImage>
void
RayConvexIntersectionImageFilter<TInputImage, TOutputImage>::BeforeThreadedGenerateData()
{
  if (this->m_ConvexShape.IsNull())
    itkExceptionMacro(<< "ConvexShape has not been set.");
}

template <class TInputImage, class TOutputImage>
void
RayConvexIntersectionImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() const
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull())
    itkExceptionMacro(<< "Geometry has not been set.");
}

template <class TInputImage, class TOutputImage>
void
RayConvexIntersectionImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread)
{
  // Iterators on input and output
  using InputRegionIterator = ProjectionsRegionConstIteratorRayBased<TInputImage>;
  InputRegionIterator * itIn = nullptr;
  itIn = InputRegionIterator::New(this->GetInput(), outputRegionForThread, m_Geometry);
  itk::ImageRegionIteratorWithIndex<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);

  // Go over each projection
  const double r = m_ConvexShape->GetDensity() / m_Attenuation;
  for (unsigned int pix = 0; pix < outputRegionForThread.GetNumberOfPixels(); pix++, itIn->Next(), ++itOut)
  {
    // Compute ray intersection length
    ConvexShape::ScalarType nearDist = NAN, farDist = NAN;
    if (m_ConvexShape->IsIntersectedByRay(itIn->GetSourcePosition(), itIn->GetDirection(), nearDist, farDist))
    {
      if (m_Attenuation == 0.)
        itOut.Set(itIn->Get() + m_ConvexShape->GetDensity() * (farDist - nearDist));
      else
        itOut.Set(itIn->Get() + r * (std::exp(m_Attenuation * farDist) - std::exp(m_Attenuation * nearDist)));
    }
    else
      itOut.Set(itIn->Get());
  }

  delete itIn;
}

} // end namespace rtk

#endif
