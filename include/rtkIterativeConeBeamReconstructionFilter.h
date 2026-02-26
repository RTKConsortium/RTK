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
#  include "rtkCudaWarpForwardProjectionImageFilter.h"
#  include "rtkCudaBackProjectionImageFilter.h"
#  include "rtkCudaWarpBackProjectionImageFilter.h"
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
class ITK_TEMPLATE_EXPORT IterativeConeBeamReconstructionFilter
  : public itk::ImageToImageFilter<TOutputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(IterativeConeBeamReconstructionFilter);

  /** Standard class type alias. */
  using Self = IterativeConeBeamReconstructionFilter;
  using Superclass = itk::ImageToImageFilter<TOutputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias */
  using VolumeType = ProjectionStackType;
  using TClipImageType = itk::Image<double, VolumeType::ImageDimension>;
  typedef enum
  {
    FP_JOSEPH = 0,
    FP_CUDARAYCAST = 2,
    FP_JOSEPHATTENUATED = 3,
    FP_ZENG = 4,
    FP_CUDAWARP = 5
  } ForwardProjectionType;
  typedef enum
  {
    BP_VOXELBASED = 0,
    BP_JOSEPH = 1,
    BP_CUDAVOXELBASED = 2,
    BP_CUDARAYCAST = 4,
    BP_JOSEPHATTENUATED = 5,
    BP_ZENG = 6,
    BP_CUDAWARP = 7
  } BackProjectionType;

  /** Typedefs of each subfilter of this composite filter */
  using ForwardProjectionFilterType = rtk::ForwardProjectionImageFilter<VolumeType, ProjectionStackType>;
  using BackProjectionFilterType = rtk::BackProjectionImageFilter<ProjectionStackType, VolumeType>;
  using ForwardProjectionPointerType = typename ForwardProjectionFilterType::Pointer;
  using BackProjectionPointerType = typename BackProjectionFilterType::Pointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(IterativeConeBeamReconstructionFilter);

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

  /** Set/Get the attenuation map for SPECT reconstruction.
   * */
  void
  SetAttenuationMap(const VolumeType * attenuationMap)
  {
    // Process object is not const-correct so the const casting is required.
    this->SetNthInput(2, const_cast<VolumeType *>(attenuationMap));
  }
  typename VolumeType::ConstPointer
  GetAttenuationMap()
  {
    return static_cast<const VolumeType *>(this->itk::ProcessObject::GetInput(2));
  }

  /** Set/Get the inferior clip image. Each pixel of the image
   ** corresponds to the value of the inferior clip of the ray
   ** emitted from that pixel. */
  void
  SetInferiorClipImage(const TClipImageType * inferiorClipImage)
  {
    // Process object is not const-correct so the const casting is required.
    this->SetInput("InferiorClipImage", const_cast<TClipImageType *>(inferiorClipImage));
  }
  typename TClipImageType::ConstPointer
  GetInferiorClipImage()
  {
    return static_cast<const TClipImageType *>(this->itk::ProcessObject::GetInput("InferiorClipImage"));
  }

  /** Set/Get the superior clip image. Each pixel of the image
   ** corresponds to the value of the superior clip of the ray
   ** emitted from that pixel. */
  void
  SetSuperiorClipImage(const TClipImageType * superiorClipImage)
  {
    // Process object is not const-correct so the const casting is required.
    this->SetInput("SuperiorClipImage", const_cast<TClipImageType *>(superiorClipImage));
  }
  typename TClipImageType::ConstPointer
  GetSuperiorClipImage()
  {
    return static_cast<const TClipImageType *>(this->itk::ProcessObject::GetInput("SuperiorClipImage"));
  }

  /** Get / Set the sigma zero of the PSF. Default is 1.5417233052142099 */
  itkGetMacro(SigmaZero, double);
  itkSetMacro(SigmaZero, double);

  /** Get / Set the alpha of the PSF. Default is 0.016241189545787734 */
  itkGetMacro(AlphaPSF, double);
  itkSetMacro(AlphaPSF, double);

  /** Get / Set step size along ray (in mm). Default is 1 mm. */
  itkGetConstMacro(StepSize, double);
  itkSetMacro(StepSize, double);

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

  /** PSF correction coefficients */
  double m_SigmaZero{ 1.5417233052142099 };
  double m_AlphaPSF{ 0.016241189545787734 };

  /** Step size along ray (in mm). */
  double m_StepSize{ 1.0 };

  /** Instantiate forward and back projectors using SFINAE. */
  using CPUImageType =
    typename itk::Image<typename ProjectionStackType::PixelType, ProjectionStackType::ImageDimension>;
  template <typename ImageType>
  using EnableCudaScalarAndVectorType = typename std::enable_if<
    !std::is_same_v<CPUImageType, ImageType> &&
    std::is_same_v<typename itk::PixelTraits<typename ImageType::PixelType>::ValueType, float> &&
    (itk::PixelTraits<typename ImageType::PixelType>::Dimension == 1 ||
     itk::PixelTraits<typename ImageType::PixelType>::Dimension == 2 ||
     itk::PixelTraits<typename ImageType::PixelType>::Dimension == 3)>::type;
  template <typename ImageType>
  using DisableCudaScalarAndVectorType = typename std::enable_if<
    std::is_same_v<CPUImageType, ImageType> ||
    !std::is_same_v<typename itk::PixelTraits<typename ImageType::PixelType>::ValueType, float> ||
    (itk::PixelTraits<typename ImageType::PixelType>::Dimension != 1 &&
     itk::PixelTraits<typename ImageType::PixelType>::Dimension != 2 &&
     itk::PixelTraits<typename ImageType::PixelType>::Dimension != 3)>::type;
  template <typename ImageType>
  using EnableCudaScalarType = typename std::enable_if<
    !std::is_same_v<CPUImageType, ImageType> &&
    std::is_same_v<typename itk::PixelTraits<typename ImageType::PixelType>::ValueType, float> &&
    itk::PixelTraits<typename ImageType::PixelType>::Dimension == 1>::type;
  template <typename ImageType>
  using DisableCudaScalarType = typename std::enable_if<
    std::is_same_v<CPUImageType, ImageType> ||
    !std::is_same_v<typename itk::PixelTraits<typename ImageType::PixelType>::ValueType, float> ||
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
    auto * cudaFw = dynamic_cast<rtk::CudaForwardProjectionImageFilter<ImageType, ImageType> *>(fw.GetPointer());
    if (cudaFw == nullptr)
    {
      itkExceptionMacro(<< "Failed to cast forward projector to CudaForwardProjectionImageFilter.");
    }
    cudaFw->SetStepSize(m_StepSize);
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


  template <typename ImageType, EnableCudaScalarType<ImageType> * = nullptr>
  ForwardProjectionPointerType
  InstantiateCudaWarpForwardProjection()
  {
    ForwardProjectionPointerType fw;
#ifdef RTK_USE_CUDA
    fw = CudaWarpForwardProjectionImageFilter::New();
    auto * cudaWarpFw = dynamic_cast<rtk::CudaWarpForwardProjectionImageFilter *>(fw.GetPointer());
    if (cudaWarpFw == nullptr)
    {
      itkExceptionMacro(<< "Failed to cast forward projector to CudaWarpForwardProjectionImageFilter.");
    }
    cudaWarpFw->SetStepSize(m_StepSize);
#endif
    return fw;
  }


  template <typename ImageType, DisableCudaScalarType<ImageType> * = nullptr>
  ForwardProjectionPointerType
  InstantiateCudaWarpForwardProjection()
  {
    itkGenericExceptionMacro(
      << "CudaWarpForwardProjectionImageFilter only available with 3D CudaImage of float or itk::Vector<float,3>.");
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
    if (this->GetAttenuationMap().IsNotNull())
    {
      fw->SetInput(2, this->GetAttenuationMap());
    }
    else
    {
      itkExceptionMacro(<< "Set Joseph attenuated forward projection filter but no attenuation map is given");
      return nullptr;
    }
    if (this->GetSuperiorClipImage().IsNotNull())
    {
      auto * josephAttenuatedForward =
        dynamic_cast<rtk::JosephForwardAttenuatedProjectionImageFilter<VolumeType, ProjectionStackType> *>(
          fw.GetPointer());
      if (josephAttenuatedForward == nullptr)
      {
        itkExceptionMacro(<< "Failed to cast forward projector to JosephForwardAttenuatedProjectionImageFilter.");
      }
      josephAttenuatedForward->SetSuperiorClipImage(this->GetSuperiorClipImage());
    }
    if (this->GetInferiorClipImage().IsNotNull())
    {
      auto * josephAttenuatedForward =
        dynamic_cast<rtk::JosephForwardAttenuatedProjectionImageFilter<VolumeType, ProjectionStackType> *>(
          fw.GetPointer());
      if (josephAttenuatedForward == nullptr)
      {
        itkExceptionMacro(<< "Failed to cast forward projector to JosephForwardAttenuatedProjectionImageFilter.");
      }
      josephAttenuatedForward->SetInferiorClipImage(this->GetInferiorClipImage());
    }
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
    if (this->GetAttenuationMap().IsNotNull())
    {
      fw->SetInput(2, this->GetAttenuationMap());
    }
    auto * zengForward =
      dynamic_cast<rtk::ZengForwardProjectionImageFilter<VolumeType, ProjectionStackType> *>(fw.GetPointer());
    if (zengForward == nullptr)
    {
      itkExceptionMacro(<< "Failed to cast forward projector to ZengForwardProjectionImageFilter.");
    }
    zengForward->SetSigmaZero(m_SigmaZero);
    zengForward->SetAlpha(m_AlphaPSF);
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
  InstantiateCudaWarpBackProjection()
  {
    BackProjectionPointerType bp;
#ifdef RTK_USE_CUDA
    bp = CudaWarpBackProjectionImageFilter::New();
#endif
    return bp;
  }


  template <typename ImageType, DisableCudaScalarType<ImageType> * = nullptr>
  BackProjectionPointerType
  InstantiateCudaWarpBackProjection()
  {
    itkGenericExceptionMacro(
      << "CudaWarpBackProjectionImageFilter only available with 3D CudaImage of float or itk::Vector<float,3>.");
    return nullptr;
  }

  template <typename ImageType, EnableCudaScalarType<ImageType> * = nullptr>
  BackProjectionPointerType
  InstantiateCudaRayCastBackProjection()
  {
    BackProjectionPointerType bp;
#ifdef RTK_USE_CUDA
    bp = CudaRayCastBackProjectionImageFilter::New();
    auto * cudaRayCastBp = dynamic_cast<rtk::CudaRayCastBackProjectionImageFilter *>(bp.GetPointer());
    if (cudaRayCastBp == nullptr)
    {
      itkExceptionMacro(<< "Failed to cast back projector to CudaRayCastBackProjectionImageFilter.");
    }
    cudaRayCastBp->SetStepSize(m_StepSize);
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
    if (this->GetAttenuationMap().IsNotNull())
    {
      bp->SetInput(2, this->GetAttenuationMap());
      return bp;
    }
    else
    {
      itkExceptionMacro(<< "Set Joseph attenuated backprojection filter but no attenuation map is given");
      return nullptr;
    }
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
    if (this->GetAttenuationMap().IsNotNull())
    {
      bp->SetInput(2, this->GetAttenuationMap());
    }
    auto * zengBack =
      dynamic_cast<rtk::ZengBackProjectionImageFilter<VolumeType, ProjectionStackType> *>(bp.GetPointer());
    if (zengBack == nullptr)
    {
      itkExceptionMacro(<< "Failed to cast back projector to ZengBackProjectionImageFilter.");
    }
    zengBack->SetSigmaZero(m_SigmaZero);
    zengBack->SetAlpha(m_AlphaPSF);
    return bp;
  }

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkIterativeConeBeamReconstructionFilter.hxx"
#endif

#endif
