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

#ifndef rtkIterativeConeBeamReconstructionFilter_h
#define rtkIterativeConeBeamReconstructionFilter_h

#include <itkPixelTraits.h>

// Forward projection filters
#include "rtkConfiguration.h"
#include "rtkJosephForwardAttenuatedProjectionImageFilter.h"
#include "rtkZengForwardProjectionImageFilter.h"
// Back projection filters
#include "rtkJosephBackAttenuatedProjectionImageFilter.h"
#include "rtkZengBackProjectionImageFilter.h"

#ifdef RTK_USE_CUDA
#  include "rtkCudaForwardProjectionImageFilter.h"
#  include "rtkCudaBackProjectionImageFilter.h"
#  include "rtkCudaRayCastBackProjectionImageFilter.h"
#endif

#include <random>
#include <algorithm>

namespace rtk
{

/** \class IterativeConeBeamReconstructionFilter
 * \brief Mother class for cone beam reconstruction filters which
 * need runtime selection of their forward and back projection filters
 *
 * IterativeConeBeamReconstructionFilter defines methods to set the forward
 * and/or back projection filter(s) of a IterativeConeBeamReconstructionFilter
 * at runtime
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */
template <class TOutputImage, class ProjectionStackType = TOutputImage>
class ITK_EXPORT IterativeConeBeamReconstructionFilter : public itk::ImageToImageFilter<TOutputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(IterativeConeBeamReconstructionFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(IterativeConeBeamReconstructionFilter);
#endif

  /** Standard class type alias. */
  using Self = IterativeConeBeamReconstructionFilter;
  using Superclass = itk::ImageToImageFilter<TOutputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias */
  using VolumeType = ProjectionStackType;
  typedef enum
  {
    FP_JOSEPH = 0,
    FP_CUDARAYCAST = 2,
    FP_JOSEPHATTENUATED = 3,
    FP_ZENG = 4
  } ForwardProjectionType;
  typedef enum
  {
    BP_VOXELBASED = 0,
    BP_JOSEPH = 1,
    BP_CUDAVOXELBASED = 2,
    BP_CUDARAYCAST = 4,
    BP_JOSEPHATTENUATED = 5,
    BP_ZENG = 6
  } BackProjectionType;

  /** Typedefs of each subfilter of this composite filter */
  using ForwardProjectionFilterType = rtk::ForwardProjectionImageFilter<VolumeType, ProjectionStackType>;
  using BackProjectionFilterType = rtk::BackProjectionImageFilter<ProjectionStackType, VolumeType>;
  using ForwardProjectionPointerType = typename ForwardProjectionFilterType::Pointer;
  using BackProjectionPointerType = typename BackProjectionFilterType::Pointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(IterativeConeBeamReconstructionFilter, itk::ImageToImageFilter);

  /** Accessors to forward and backprojection types. */
  virtual void
  SetForwardProjectionFilter(ForwardProjectionType fwtype);
  ForwardProjectionType
  GetForwardProjectionFilter()
  {
    return m_CurrentForwardProjectionConfiguration;
  }
  virtual void
  SetBackProjectionFilter(BackProjectionType bptype);
  BackProjectionType
  GetBackProjectionFilter()
  {
    return m_CurrentBackProjectionConfiguration;
  }

protected:
  IterativeConeBeamReconstructionFilter();
  ~IterativeConeBeamReconstructionFilter() override = default;

  /** Creates and returns an instance of the back projection filter.
   * To be used in SetBackProjectionFilter. */
  virtual BackProjectionPointerType
  InstantiateBackProjectionFilter(int bptype);

  /** Creates and returns an instance of the forward projection filter.
   * To be used in SetForwardProjectionFilter. */
  virtual ForwardProjectionPointerType
  InstantiateForwardProjectionFilter(int fwtype);

  /** Internal variables storing the current forward
    and back projection methods */
  ForwardProjectionType m_CurrentForwardProjectionConfiguration;
  BackProjectionType    m_CurrentBackProjectionConfiguration;

  /** A random generating engine is needed to use the C++17 comliant code for std::shuffle.
   */
  std::default_random_engine m_DefaultRandomEngine = std::default_random_engine{};

  /** Instantiate forward and back projectors using SFINAE. */
  using CPUImageType =
    typename itk::Image<typename ProjectionStackType::PixelType, ProjectionStackType::ImageDimension>;
  template <typename ImageType>
  using EnableCudaScalarAndVectorType = typename std::enable_if<
    !std::is_same<CPUImageType, ImageType>::value &&
    std::is_same<typename itk::PixelTraits<typename ImageType::PixelType>::ValueType, float>::value &&
    (itk::PixelTraits<typename ImageType::PixelType>::Dimension == 1 ||
     itk::PixelTraits<typename ImageType::PixelType>::Dimension == 2 ||
     itk::PixelTraits<typename ImageType::PixelType>::Dimension == 3)>::type;
  template <typename ImageType>
  using DisableCudaScalarAndVectorType = typename std::enable_if<
    std::is_same<CPUImageType, ImageType>::value ||
    !std::is_same<typename itk::PixelTraits<typename ImageType::PixelType>::ValueType, float>::value ||
    (itk::PixelTraits<typename ImageType::PixelType>::Dimension != 1 &&
     itk::PixelTraits<typename ImageType::PixelType>::Dimension != 2 &&
     itk::PixelTraits<typename ImageType::PixelType>::Dimension != 3)>::type;
  template <typename ImageType>
  using EnableCudaScalarType = typename std::enable_if<
    !std::is_same<CPUImageType, ImageType>::value &&
    std::is_same<typename itk::PixelTraits<typename ImageType::PixelType>::ValueType, float>::value &&
    itk::PixelTraits<typename ImageType::PixelType>::Dimension == 1>::type;
  template <typename ImageType>
  using DisableCudaScalarType = typename std::enable_if<
    std::is_same<CPUImageType, ImageType>::value ||
    !std::is_same<typename itk::PixelTraits<typename ImageType::PixelType>::ValueType, float>::value ||
    itk::PixelTraits<typename ImageType::PixelType>::Dimension != 1>::type;
  template <typename ImageType>
  using EnableVectorType =
    typename std::enable_if<itk::PixelTraits<typename ImageType::PixelType>::Dimension != 1>::type;
  template <typename ImageType>
  using DisableVectorType =
    typename std::enable_if<itk::PixelTraits<typename ImageType::PixelType>::Dimension == 1>::type;

  template <typename ImageType, EnableCudaScalarAndVectorType<ImageType> * = nullptr>
  ForwardProjectionPointerType
  InstantiateCudaForwardProjection()
  {
    ForwardProjectionPointerType fw;
#ifdef RTK_USE_CUDA
    fw = CudaForwardProjectionImageFilter<ImageType, ImageType>::New();
#endif
    return fw;
  }


  template <typename ImageType, DisableCudaScalarAndVectorType<ImageType> * = nullptr>
  ForwardProjectionPointerType
  InstantiateCudaForwardProjection()
  {
    itkGenericExceptionMacro(
      << "CudaRayCastBackProjectionImageFilter only available with 3D CudaImage of float or itk::Vector<float,3>.");
    return nullptr;
  }


  template <typename ImageType, EnableVectorType<ImageType> * = nullptr>
  ForwardProjectionPointerType
  InstantiateJosephForwardAttenuatedProjection()
  {
    itkGenericExceptionMacro(<< "JosephForwardAttenuatedProjectionImageFilter only available with scalar pixel types.");
    return nullptr;
  }


  template <typename ImageType, DisableVectorType<ImageType> * = nullptr>
  ForwardProjectionPointerType
  InstantiateJosephForwardAttenuatedProjection()
  {
    ForwardProjectionPointerType fw;
    fw = JosephForwardAttenuatedProjectionImageFilter<VolumeType, ProjectionStackType>::New();
    return fw;
  }

  template <typename ImageType, EnableVectorType<ImageType> * = nullptr>
  ForwardProjectionPointerType
  InstantiateZengForwardProjection()
  {
    itkGenericExceptionMacro(<< "JosephForwardAttenuatedProjectionImageFilter only available with scalar pixel types.");
    return nullptr;
  }


  template <typename ImageType, DisableVectorType<ImageType> * = nullptr>
  ForwardProjectionPointerType
  InstantiateZengForwardProjection()
  {
    ForwardProjectionPointerType fw;
    fw = ZengForwardProjectionImageFilter<VolumeType, ProjectionStackType>::New();
    return fw;
  }

  template <typename ImageType, EnableCudaScalarAndVectorType<ImageType> * = nullptr>
  BackProjectionPointerType
  InstantiateCudaBackProjection()
  {
    BackProjectionPointerType bp;
#ifdef RTK_USE_CUDA
    bp = CudaBackProjectionImageFilter<ImageType>::New();
#endif
    return bp;
  }


  template <typename ImageType, DisableCudaScalarAndVectorType<ImageType> * = nullptr>
  BackProjectionPointerType
  InstantiateCudaBackProjection()
  {
    itkGenericExceptionMacro(
      << "CudaBackProjectionImageFilter only available with 3D CudaImage of float or itk::Vector<float,3>.");
    return nullptr;
  }


  template <typename ImageType, EnableCudaScalarType<ImageType> * = nullptr>
  BackProjectionPointerType
  InstantiateCudaRayCastBackProjection()
  {
    BackProjectionPointerType bp;
#ifdef RTK_USE_CUDA
    bp = CudaRayCastBackProjectionImageFilter::New();
#endif
    return bp;
  }


  template <typename ImageType, DisableCudaScalarType<ImageType> * = nullptr>
  BackProjectionPointerType
  InstantiateCudaRayCastBackProjection()
  {
    itkGenericExceptionMacro(<< "CudaRayCastBackProjectionImageFilter only available with 3D CudaImage of float.");
    return nullptr;
  }


  template <typename ImageType, EnableVectorType<ImageType> * = nullptr>
  BackProjectionPointerType
  InstantiateJosephBackAttenuatedProjection()
  {
    itkGenericExceptionMacro(<< "JosephBackAttenuatedProjectionImageFilter only available with scalar pixel types.");
    return nullptr;
  }


  template <typename ImageType, DisableVectorType<ImageType> * = nullptr>
  BackProjectionPointerType
  InstantiateJosephBackAttenuatedProjection()
  {
    BackProjectionPointerType bp;
    bp = JosephBackAttenuatedProjectionImageFilter<ImageType, ImageType>::New();
    return bp;
  }

  template <typename ImageType, EnableVectorType<ImageType> * = nullptr>
  BackProjectionPointerType
  InstantiateZengBackProjection()
  {
    itkGenericExceptionMacro(<< "JosephBackAttenuatedProjectionImageFilter only available with scalar pixel types.");
    return nullptr;
  }


  template <typename ImageType, DisableVectorType<ImageType> * = nullptr>
  BackProjectionPointerType
  InstantiateZengBackProjection()
  {
    BackProjectionPointerType bp;
    bp = ZengBackProjectionImageFilter<ImageType, ImageType>::New();
    return bp;
  }

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkIterativeConeBeamReconstructionFilter.hxx"
#endif

#endif
