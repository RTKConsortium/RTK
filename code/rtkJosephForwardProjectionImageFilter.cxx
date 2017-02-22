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

#include "rtkJosephForwardProjectionImageFilter.h"

namespace rtk
{

template<>
void
rtk::JosephForwardProjectionImageFilter<itk::VectorImage<float, 3>,
                                        itk::VectorImage<float, 3>,
                                        Functor::VectorInterpolationWeightMultiplication<float, double, itk::VariableLengthVector<float>>,
                                        Functor::VectorProjectedValueAccumulation<itk::VariableLengthVector<float>, itk::VariableLengthVector<float> > >
::Accumulate(ThreadIdType threadId,
             rtk::ProjectionsRegionConstIteratorRayBased<itk::VectorImage<float, 3> >* itIn,
             itk::ImageRegionIteratorWithIndex<itk::VectorImage<float, 3> > itOut,
             itk::VariableLengthVector<float> sum,
             rtk::RayBoxIntersectionFunction<double, 3>::VectorType stepMM,
             rtk::RayBoxIntersectionFunction<double, 3>::VectorType sourcePosition,
             rtk::RayBoxIntersectionFunction<double, 3>::VectorType dirVox,
             rtk::RayBoxIntersectionFunction<double, 3>::VectorType np,
             rtk::RayBoxIntersectionFunction<double, 3>::VectorType fp)
{
    itOut.Set(m_ProjectedValueAccumulation(threadId,
                                           itIn->Get(),
                                           itOut.Get(),
                                           sum,
                                           stepMM,
                                           sourcePosition,
                                           dirVox,
                                           np,
                                           fp));
}

template <>
itk::VariableLengthVector<float>
rtk::JosephForwardProjectionImageFilter<itk::VectorImage<float, 3>,
                                        itk::VectorImage<float, 3>,
                                        Functor::VectorInterpolationWeightMultiplication<float, double, itk::VariableLengthVector<float>>,
                                        Functor::VectorProjectedValueAccumulation<itk::VariableLengthVector<float>, itk::VariableLengthVector<float> > >
::FillPixel(float value)
{
itk::VariableLengthVector<float> vect;
vect.SetSize(this->GetInput(1)->GetNumberOfComponentsPerPixel());
vect.Fill(value);
return (vect);
}

} // end namespace rtk
