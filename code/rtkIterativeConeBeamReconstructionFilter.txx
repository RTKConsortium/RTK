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

#ifndef __rtkIterativeConeBeamReconstructionFilter_txx
#define __rtkIterativeConeBeamReconstructionFilter_txx

#include "rtkIterativeConeBeamReconstructionFilter.h"

namespace rtk
{
  template<class TInputImage, class TOutputImage>
  IterativeConeBeamReconstructionFilter<TInputImage, TOutputImage>
  ::IterativeConeBeamReconstructionFilter()
  {

  }

  template<class TInputImage, class TOutputImage>
  typename IterativeConeBeamReconstructionFilter<TInputImage, TOutputImage>::ForwardProjectionPointerType
  IterativeConeBeamReconstructionFilter<TInputImage, TOutputImage>
  ::InstantiateForwardProjectionFilter (int fwtype)
  {
    ForwardProjectionPointerType fw;
    switch(fwtype)
      {
      case(0):
        fw = rtk::JosephForwardProjectionImageFilter<TOutputImage, TOutputImage>::New();
      break;
      case(1):
        fw = rtk::RayCastInterpolatorForwardProjectionImageFilter<TOutputImage, TOutputImage>::New();
      break;
      case(2):
      #if CUDA_FOUND
        fw = rtk::CudaForwardProjectionImageFilter::New();
      #else
        std::cerr << "The program has not been compiled with cuda option" << std::endl;
      #endif
      break;
      case(3):
        fw = rtk::SiddonForwardProjectionImageFilter<TOutputImage, TOutputImage>::New();
      break;

      default:
        std::cerr << "Unhandled --method value." << std::endl;
      }
    return fw;
  }

  template<class TInputImage, class TOutputImage>
  typename IterativeConeBeamReconstructionFilter<TInputImage, TOutputImage>::BackProjectionPointerType
  IterativeConeBeamReconstructionFilter<TInputImage, TOutputImage>::InstantiateBackProjectionFilter(int bptype)
  {
    BackProjectionPointerType bp;
    switch(bptype)
      {
      case(0):
        bp = rtk::BackProjectionImageFilter<TOutputImage, TOutputImage>::New();
        break;
      case(1):
        bp = rtk::JosephBackProjectionImageFilter<TOutputImage, TOutputImage>::New();
        break;
      case(2):
      #if CUDA_FOUND
        bp = rtk::CudaBackProjectionImageFilter::New();
      #else
        std::cerr << "The program has not been compiled with cuda option" << std::endl;
      #endif
      break;
      case(3):
        bp = rtk::NormalizedJosephBackProjectionImageFilter<TOutputImage, TOutputImage>::New();
        break;

      default:
        std::cerr << "Unhandled --bp value." << std::endl;
      }
    return bp;
  }
} // end namespace rtk

#endif // __rtkIterativeConeBeamReconstructionFilter_txx
