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

#ifdef RTK_USE_CUDA
  template<> inline
  typename IterativeConeBeamReconstructionFilter<itk::CudaImage<float, 4>, itk::CudaImage<float, 3>>::ForwardProjectionPointerType
  IterativeConeBeamReconstructionFilter<itk::CudaImage<float, 4>, itk::CudaImage<float, 3>>
  ::InstantiateForwardProjectionFilter (int fwtype)
  {
    ForwardProjectionPointerType fw;
    switch(fwtype)
      {
      case(FP_JOSEPH):
        fw = rtk::JosephForwardProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>::New();
      break;
      case(FP_CUDARAYCAST):
        fw = rtk::CudaForwardProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>::New();
      break;

      default:
        itkGenericExceptionMacro(<< "Unhandled --fp value.");
      }
    return fw;
  }

  template<> inline
  typename IterativeConeBeamReconstructionFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>::ForwardProjectionPointerType
  IterativeConeBeamReconstructionFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>
  ::InstantiateForwardProjectionFilter (int fwtype)
  {
    ForwardProjectionPointerType fw;
    switch(fwtype)
      {
      case(FP_JOSEPH):
        fw = rtk::JosephForwardProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>::New();
      break;
      case(FP_CUDARAYCAST):
        fw = rtk::CudaForwardProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>::New();
      break;
      case(FP_JOSEPHATTENUATED):
    {
        fw = rtk::JosephForwardAttenuatedProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>::New();
        m_CurrentForwardProjectionConfiguration = FP_JOSEPHATTENUATED;
      }
      break;
      default:
        itkGenericExceptionMacro(<< "Unhandled --fp value.");
      }
    return fw;
  }

  template<> inline
  typename IterativeConeBeamReconstructionFilter<itk::CudaImage<itk::Vector<float, 3>, 3>, itk::CudaImage<itk::Vector<float, 3>, 3>>::ForwardProjectionPointerType
  IterativeConeBeamReconstructionFilter<itk::CudaImage<itk::Vector<float, 3>, 3>, itk::CudaImage<itk::Vector<float, 3>, 3>>
  ::InstantiateForwardProjectionFilter (int fwtype)
  {
    ForwardProjectionPointerType fw;
    switch(fwtype)
      {
      case(FP_JOSEPH):
        fw = rtk::JosephForwardProjectionImageFilter<itk::CudaImage<itk::Vector<float, 3>, 3>, itk::CudaImage<itk::Vector<float, 3>, 3>>::New();
      break;
      case(FP_CUDARAYCAST):
        fw = rtk::CudaForwardProjectionImageFilter<itk::CudaImage<itk::Vector<float, 3>, 3>, itk::CudaImage<itk::Vector<float, 3>, 3>>::New();
      break;
      default:
        itkGenericExceptionMacro(<< "Unhandled --fp value.");
      }
    return fw;
  }
#endif

  template<> inline
  typename IterativeConeBeamReconstructionFilter<itk::Image<itk::Vector<float, 3>, 3>, itk::Image<itk::Vector<float, 3>, 3>>::ForwardProjectionPointerType
  IterativeConeBeamReconstructionFilter<itk::Image<itk::Vector<float, 3>, 3>, itk::Image<itk::Vector<float, 3>, 3>>
  ::InstantiateForwardProjectionFilter (int fwtype)
  {
    ForwardProjectionPointerType fw;
    switch(fwtype)
      {
      case(FP_JOSEPH):
        fw = rtk::JosephForwardProjectionImageFilter<itk::Image<itk::Vector<float, 3>, 3>, itk::Image<itk::Vector<float, 3>, 3>>::New();
      break;
      default:
        itkGenericExceptionMacro(<< "Unhandled --fp value.");
      }
    return fw;
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
        itkGenericExceptionMacro(<< "The CUDA forward projector currently accepts 3D images of floats or of itk::Vector<float, 3>");
      #else
        itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
      #endif
      break;
      case(FP_JOSEPHATTENUATED):
          fw = rtk::JosephForwardAttenuatedProjectionImageFilter<VolumeType, ProjectionStackType>::New();
      break;
      default:
        itkGenericExceptionMacro(<< "Unhandled --fp value.");
      }
    return fw;
  }

#ifdef RTK_USE_CUDA
  template<> inline
  typename IterativeConeBeamReconstructionFilter<itk::CudaImage<float, 4>, itk::CudaImage<float, 3> >::BackProjectionPointerType
  IterativeConeBeamReconstructionFilter<itk::CudaImage<float, 4>, itk::CudaImage<float, 3> >::InstantiateBackProjectionFilter(int bptype)
  {
    BackProjectionPointerType bp;
    switch(bptype)
      {
      case(BP_VOXELBASED):
        bp = rtk::BackProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3> >::New();
        break;
      case(BP_JOSEPH):
        bp = rtk::JosephBackProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3> >::New();
        break;
      case(BP_CUDAVOXELBASED):
        bp = rtk::CudaBackProjectionImageFilter<itk::CudaImage<float, 3>>::New();
      break;
      case(BP_CUDARAYCAST):
        bp = rtk::CudaRayCastBackProjectionImageFilter::New();
        break;
      default:
        itkGenericExceptionMacro(<< "Unhandled --bp value.");
      }
    return bp;
  }

  template<> inline
  typename IterativeConeBeamReconstructionFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3> >::BackProjectionPointerType
  IterativeConeBeamReconstructionFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3> >::InstantiateBackProjectionFilter(int bptype)
  {
    BackProjectionPointerType bp;
    switch(bptype)
      {
      case(BP_VOXELBASED):
        bp = rtk::BackProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3> >::New();
        break;
      case(BP_JOSEPH):
        bp = rtk::JosephBackProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3> >::New();
        break;
      case(BP_CUDAVOXELBASED):
        bp = rtk::CudaBackProjectionImageFilter<itk::CudaImage<float, 3>>::New();
      break;
      case(BP_CUDARAYCAST):
        bp = rtk::CudaRayCastBackProjectionImageFilter::New();
        break;
      case(BP_JOSEPHATTENUATED):
    {
        bp = rtk::JosephBackAttenuatedProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>::New();
         m_CurrentBackProjectionConfiguration =BP_JOSEPHATTENUATED;
    }
        break;
      default:
        itkGenericExceptionMacro(<< "Unhandled --bp value.");
      }
    return bp;
  }

  template<> inline
  typename IterativeConeBeamReconstructionFilter<itk::CudaImage<itk::Vector<float, 3>, 3>, itk::CudaImage<itk::Vector<float, 3>, 3> >::BackProjectionPointerType
  IterativeConeBeamReconstructionFilter<itk::CudaImage<itk::Vector<float, 3>, 3>, itk::CudaImage<itk::Vector<float, 3>, 3> >::InstantiateBackProjectionFilter(int bptype)
  {
    BackProjectionPointerType bp;
    switch(bptype)
      {
      case(BP_VOXELBASED):
        bp = rtk::BackProjectionImageFilter<itk::CudaImage<itk::Vector<float, 3>, 3>, itk::CudaImage<itk::Vector<float, 3>, 3> >::New();
        break;
      case(BP_JOSEPH):
        bp = rtk::JosephBackProjectionImageFilter<itk::CudaImage<itk::Vector<float, 3>, 3>, itk::CudaImage<itk::Vector<float, 3>, 3> >::New();
        break;
      case(BP_CUDAVOXELBASED):
        bp = rtk::CudaBackProjectionImageFilter<itk::CudaImage<itk::Vector<float, 3>, 3>>::New();
      break;
      case(BP_CUDARAYCAST):
        itkGenericExceptionMacro(<< "The CUDA ray cast back projector does not handle images of itk::Vector<float, 3> yet");
        break;
      default:
        itkGenericExceptionMacro(<< "Unhandled --bp value.");
      }
    return bp;
  }
#endif

  template<> inline
  typename IterativeConeBeamReconstructionFilter<itk::Image<itk::Vector<float, 3>, 3>, itk::Image<itk::Vector<float, 3>, 3> >::BackProjectionPointerType
  IterativeConeBeamReconstructionFilter<itk::Image<itk::Vector<float, 3>, 3>, itk::Image<itk::Vector<float, 3>, 3> >::InstantiateBackProjectionFilter(int bptype)
  {
    BackProjectionPointerType bp;
    switch(bptype)
      {
      case(BP_VOXELBASED):
        bp = rtk::BackProjectionImageFilter<itk::Image<itk::Vector<float, 3>, 3>, itk::Image<itk::Vector<float, 3>, 3> >::New();
        break;
      case(BP_JOSEPH):
        bp = rtk::JosephBackProjectionImageFilter<itk::Image<itk::Vector<float, 3>, 3>, itk::Image<itk::Vector<float, 3>, 3> >::New();
        break;
      default:
        itkGenericExceptionMacro(<< "Unhandled --bp value.");
      }
    return bp;
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
        itkGenericExceptionMacro(<< "The CUDA voxel based back projector currently accepts 3D images of floats or of itk::Vector<float, 3>");
      #else
        itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
      #endif
      break;
      case(BP_CUDARAYCAST):
      #ifdef RTK_USE_CUDA
        itkGenericExceptionMacro(<< "The CUDA ray cast back projector currently accepts 3D float images only");
      #else
        itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
      #endif
        break;
      case(BP_JOSEPHATTENUATED):
        bp = rtk::JosephBackAttenuatedProjectionImageFilter<ProjectionStackType, VolumeType>::New();
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
