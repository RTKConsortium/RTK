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

#include "rtkIterativeConeBeamReconstructionFilter.h"

namespace rtk
{


template<>
typename rtk::BackProjectionImageFilter<itk::VectorImage<float, 3>,
itk::VectorImage<float, 3>,
itk::VectorImage<float, 2> >::Pointer
IterativeConeBeamReconstructionFilter< itk::VectorImage<float, 3>, itk::VectorImage<float, 3>, itk::VectorImage<float, 2> >
::InstantiateBackProjectionFilter(int bptype)
{
  rtk::BackProjectionImageFilter< itk::VectorImage<float, 3>, itk::VectorImage<float, 3>, itk::VectorImage<float, 2> >::Pointer bp;
  switch(bptype)
    {
//    case(0):
//      bp = rtk::BackProjectionImageFilter<ProjectionStackType, VolumeType>::New();
//      break;
    case(1):
      bp = rtk::JosephBackProjectionImageFilter<itk::VectorImage<float, 3>,
              itk::VectorImage<float, 3>,
              Functor::SplatWeightMultiplication< float, double, float >,
              itk::VectorImage<float, 2> >::New();
      break;
//    case(2):
//    #ifdef RTK_USE_CUDA
//      bp = rtk::CudaBackProjectionImageFilter::New();
//    #else
//      itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
//    #endif
//    break;
//    case(3):
//      bp = rtk::NormalizedJosephBackProjectionImageFilter<ProjectionStackType, VolumeType>::New();
//      break;
//    case(4):
//    #ifdef RTK_USE_CUDA
//      bp = rtk::CudaRayCastBackProjectionImageFilter::New();
//    #else
//      itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
//    #endif
//      break;
    default:
      itkGenericExceptionMacro(<< "Unhandled --bp value.");
    }
  return bp;
}

template<>
typename rtk::ForwardProjectionImageFilter<itk::VectorImage<float, 3>, itk::VectorImage<float, 3> >::Pointer
IterativeConeBeamReconstructionFilter< itk::VectorImage<float, 3>, itk::VectorImage<float, 3>, itk::VectorImage<float, 2> >
::InstantiateForwardProjectionFilter(int fwtype)
{
  rtk::ForwardProjectionImageFilter<itk::VectorImage<float, 3>, itk::VectorImage<float, 3> >::Pointer fw;
  switch(fwtype)
    {
    case(0):
      fw = rtk::JosephForwardProjectionImageFilter<itk::VectorImage<float, 3>,
              itk::VectorImage<float, 3>,
              Functor::VectorInterpolationWeightMultiplication<float, double, itk::VariableLengthVector<float>>,
              Functor::VectorProjectedValueAccumulation<itk::VariableLengthVector<float>, itk::VariableLengthVector<float> > >::New();
    break;
//    case(1):
//      fw = rtk::RayCastInterpolatorForwardProjectionImageFilter<VolumeType, ProjectionStackType>::New();
//    break;
//    case(2):
//    #ifdef RTK_USE_CUDA
//      fw = rtk::CudaForwardProjectionImageFilter<VolumeType, ProjectionStackType>::New();
//    #else
//      itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
//    #endif
//    break;

    default:
      itkGenericExceptionMacro(<< "Unhandled --fp value.");
    }
  return fw;
}


} // end namespace rtk
