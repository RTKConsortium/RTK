/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *
 *=========================================================================*/
#ifndef rtkCudaZengForwardProjectionImageFilter_h
#define rtkCudaZengForwardProjectionImageFilter_h

#include "rtkConfiguration.h"
#ifdef RTK_USE_CUDA

#  include "rtkForwardProjectionImageFilter.h"
#  include "RTKExport.h"
#  include <itkCudaImage.h>
#  include <itkCudaInPlaceImageFilter.h>
#  include <itkImageBase.h>

namespace rtk
{

/** \class CudaZengForwardProjectionImageFilter
 * \brief CUDA implementation of the rotation-based Zeng forward projector.
 *
 * The implementation reproduces the slice recursion of
 * ZengForwardProjectionImageFilter, including the depth-dependent Gaussian
 * PSF and the optional attenuation map (input 2).
 *
 * \ingroup RTK Projector CudaImageToImageFilter
 */
class RTK_EXPORT CudaZengForwardProjectionImageFilter
  : public itk::CudaInPlaceImageFilter<
      itk::CudaImage<float, 3>,
      itk::CudaImage<float, 3>,
      ForwardProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaZengForwardProjectionImageFilter);
  using ImageType = itk::CudaImage<float, 3>;
  // Metadata-only image used to build the rotated-grid transforms.  Using
  // ImageBase avoids the CUDA image factory replacing itk::Image::New().
  using CPUImageType = itk::ImageBase<3>;
  using ProjectorType = ForwardProjectionImageFilter<ImageType, ImageType>;
  using Self = CudaZengForwardProjectionImageFilter;
  using Superclass = itk::CudaInPlaceImageFilter<ImageType, ImageType, ProjectorType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaZengForwardProjectionImageFilter);

  itkGetConstMacro(SigmaZero, double);
  itkSetMacro(SigmaZero, double);
  itkGetConstMacro(Alpha, double);
  itkSetMacro(Alpha, double);

protected:
  CudaZengForwardProjectionImageFilter();
  ~CudaZengForwardProjectionImageFilter() override;
  void GPUGenerateData() override;

private:
  double m_SigmaZero{ 1.5417233052142099 };
  double m_Alpha{ 0.016241189545787734 };
  void * m_CudaWorkspace{ nullptr };
};

} // namespace rtk
#endif
#endif
