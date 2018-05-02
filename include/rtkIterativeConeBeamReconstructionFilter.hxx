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

#ifndef rtkIterativeConeBeamReconstructionFilter_hxx
#define rtkIterativeConeBeamReconstructionFilter_hxx

#include "rtkIterativeConeBeamReconstructionFilter.h"

namespace rtk
{
  template<class TOutputImage, class ProjectionStackType>
  IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType>
  ::IterativeConeBeamReconstructionFilter()
  {
    m_CurrentForwardProjectionConfiguration = FP_UNKNOWN;
    m_CurrentBackProjectionConfiguration = BP_UNKNOWN;
  }

  template<class TOutputImage, class ProjectionStackType>
  typename IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType>::ForwardProjectionPointerType
  IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType>
  ::InstantiateForwardProjectionFilter (int fwtype)
  {
    ForwardProjectionPointerType fw;
    switch(fwtype)
      {
      case(FP_JOSEPH):
        fw = rtk::JosephForwardProjectionImageFilter<VolumeType, ProjectionStackType>::New();
      break;
      case(FP_CUDARAYCAST):
      #ifdef RTK_USE_CUDA
        fw = rtk::CudaForwardProjectionImageFilter<VolumeType, ProjectionStackType>::New();
      #else
        itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
      #endif
      break;

      default:
        itkGenericExceptionMacro(<< "Unhandled --fp value.");
      }
    return fw;
  }

  template<class TOutputImage, class ProjectionStackType>
  typename IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType>::BackProjectionPointerType
  IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType>::InstantiateBackProjectionFilter(int bptype)
  {
    BackProjectionPointerType bp;
    switch(bptype)
      {
      case(BP_VOXELBASED):
        bp = rtk::BackProjectionImageFilter<ProjectionStackType, VolumeType>::New();
        break;
      case(BP_JOSEPH):
        bp = rtk::JosephBackProjectionImageFilter<ProjectionStackType, VolumeType>::New();
        break;
      case(BP_CUDAVOXELBASED):
      #ifdef RTK_USE_CUDA
        bp = rtk::CudaBackProjectionImageFilter::New();
      #else
        itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
      #endif
      break;
      case(BP_CUDARAYCAST):
      #ifdef RTK_USE_CUDA
        bp = rtk::CudaRayCastBackProjectionImageFilter::New();
      #else
        itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
      #endif
        break;
      default:
        itkGenericExceptionMacro(<< "Unhandled --bp value.");
      }
    return bp;
  }

  template<class TOutputImage, class ProjectionStackType>
  void
  IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType>
  ::SetForwardProjectionFilter (ForwardProjectionType fwtype)
  {
    if (m_CurrentForwardProjectionConfiguration != fwtype)
      {
      this->Modified();
      }
  }

  template<class TOutputImage, class ProjectionStackType>
  void
  IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType>
  ::SetBackProjectionFilter (BackProjectionType bptype)
  {
    if (m_CurrentBackProjectionConfiguration != bptype)
      {
      this->Modified();
      }
  }

} // end namespace rtk

#endif // rtkIterativeConeBeamReconstructionFilter_hxx
