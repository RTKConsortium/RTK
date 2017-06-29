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
typename rtk::BackProjectionImageFilter<itk::VectorImage<double, 3>, itk::VectorImage<double, 3> >::Pointer
IterativeConeBeamReconstructionFilter< itk::VectorImage<double, 3>, itk::VectorImage<double, 3> >
::InstantiateBackProjectionFilter(int bptype)
{
  rtk::BackProjectionImageFilter< itk::VectorImage<double, 3>, itk::VectorImage<double, 3> >::Pointer bp;
  switch(bptype)
    {
//    case(0):
//      bp = rtk::BackProjectionImageFilter<ProjectionStackType, VolumeType>::New();
//      break;
    case(1):
      bp = rtk::JosephBackProjectionImageFilter<itk::VectorImage<double, 3>,
              itk::VectorImage<double, 3>,
              Functor::SplatWeightMultiplication< double, double, double > >::New();
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
typename rtk::ForwardProjectionImageFilter<itk::VectorImage<double, 3>, itk::VectorImage<double, 3> >::Pointer
IterativeConeBeamReconstructionFilter< itk::VectorImage<double, 3>, itk::VectorImage<double, 3> >
::InstantiateForwardProjectionFilter(int fwtype)
{
  rtk::ForwardProjectionImageFilter<itk::VectorImage<double, 3>, itk::VectorImage<double, 3> >::Pointer fw;
  switch(fwtype)
    {
    case(0):
      fw = rtk::JosephForwardProjectionImageFilter<itk::VectorImage<double, 3>,
              itk::VectorImage<double, 3>,
              Functor::VectorInterpolationWeightMultiplication<double, double, itk::VariableLengthVector<double>>,
              Functor::VectorProjectedValueAccumulation<itk::VariableLengthVector<double>, itk::VariableLengthVector<double> > >::New();
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
