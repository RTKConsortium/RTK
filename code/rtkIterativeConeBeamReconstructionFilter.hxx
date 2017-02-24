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
  template<class TOutputImage, class ProjectionStackType, class TProjectionImage>
  IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType, TProjectionImage>
  ::IterativeConeBeamReconstructionFilter()
  {
    m_CurrentForwardProjectionConfiguration = -1;
    m_CurrentBackProjectionConfiguration = -1;
  }

  template<class TOutputImage, class ProjectionStackType, class TProjectionImage>
  typename IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType, TProjectionImage>::ForwardProjectionPointerType
  IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType, TProjectionImage>
  ::InstantiateForwardProjectionFilter (int fwtype)
  {
    ForwardProjectionPointerType fw;
    switch(fwtype)
      {
      case(0):
        fw = rtk::JosephForwardProjectionImageFilter<VolumeType, ProjectionStackType>::New();
      break;
      case(1):
        fw = rtk::RayCastInterpolatorForwardProjectionImageFilter<VolumeType, ProjectionStackType>::New();
      break;
      case(2):
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

  template<class TOutputImage, class ProjectionStackType, class TProjectionImage>
  typename IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType, TProjectionImage>::BackProjectionPointerType
  IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType, TProjectionImage>::InstantiateBackProjectionFilter(int bptype)
  {
    BackProjectionPointerType bp;
    switch(bptype)
      {
      case(0):
        bp = rtk::BackProjectionImageFilter<ProjectionStackType, VolumeType>::New();
        break;
      case(1):
        bp = rtk::JosephBackProjectionImageFilter<ProjectionStackType, VolumeType>::New();
        break;
      case(2):
      #ifdef RTK_USE_CUDA
        bp = rtk::CudaBackProjectionImageFilter::New();
      #else
        itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
      #endif
      break;
      case(3):
        bp = rtk::NormalizedJosephBackProjectionImageFilter<ProjectionStackType, VolumeType>::New();
        break;
      case(4):
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

  template<class TOutputImage, class ProjectionStackType, class TProjectionImage>
  void
  IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType, TProjectionImage>
  ::SetForwardProjectionFilter (int fwtype)
  {
    if (m_CurrentForwardProjectionConfiguration != fwtype)
      {
      this->Modified();
      }
  }

  template<class TOutputImage, class ProjectionStackType, class TProjectionImage>
  void
  IterativeConeBeamReconstructionFilter<TOutputImage, ProjectionStackType, TProjectionImage>
  ::SetBackProjectionFilter (int bptype)
  {
    if (m_CurrentBackProjectionConfiguration != bptype)
      {
      this->Modified();
      }
  }

} // end namespace rtk

#endif // rtkIterativeConeBeamReconstructionFilter_hxx
