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

#ifndef rtkZengForwardProjectionImageFilter_h
#define rtkZengForwardProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkMacro.h"
#include <itkPixelTraits.h>

#include <itkMultiplyImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkVector.h>
#include <itkCenteredEuler3DTransform.h>
#include <itkChangeInformationImageFilter.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkExpImageFilter.h>
#include <itkConstantBoundaryCondition.h>

#include <itkVectorImage.h>
namespace rtk
{

/** \class ZengForwardProjectionImageFilter
 * \brief Zeng forward projection.
 *
 * Performs a rotation based forward projection, i.e. the volume is rotated so that
 * the face of the image volume is parallel to the detector and the projection is done
 * by summing the collumn. This projector is used to perform the PSF correction
 * describe in [Zeng 1996]. The forward projector tests if the detector
 * has been placed after the source and the volume. If the detector is in the volume
 * the sum is performed only until that point.
 *
 * \test rtkZengforwardprojectiontest.cxx
 *
 * \author Antoine Robert
 *
 * \ingroup RTK Projector
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT ZengForwardProjectionImageFilter : public ForwardProjectionImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class usings. */
  using VectorType = itk::Vector<double, 3>;
  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using InputCPUImageType = itk::Image<InputPixelType, 3>;
  using OuputCPUImageType = itk::Image<OutputPixelType, 3>;
  using PointType = typename InputCPUImageType::PointType;

  using Self = ZengForwardProjectionImageFilter;
  using Superclass = ForwardProjectionImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using AddImageFilterType = itk::AddImageFilter<OuputCPUImageType, OuputCPUImageType>;
  using AddImageFilterPointerType = typename AddImageFilterType::Pointer;
  using PasteImageFilterType = itk::PasteImageFilter<OuputCPUImageType, InputCPUImageType>;
  using PasteImageFilterPointerType = typename PasteImageFilterType::Pointer;
  using DiscreteGaussianFilterType = itk::DiscreteGaussianImageFilter<OuputCPUImageType, OuputCPUImageType>;
  using DiscreteGaussianFilterPointeurType = typename DiscreteGaussianFilterType::Pointer;
  using ResampleImageFilterType = itk::ResampleImageFilter<OuputCPUImageType, OuputCPUImageType>;
  using ResampleImageFilterPointerType = typename ResampleImageFilterType::Pointer;
  using TransformType = itk::CenteredEuler3DTransform<double>;
  using TransformPointerType = typename TransformType::Pointer;
  using ChangeInformationFilterType = itk::ChangeInformationImageFilter<OuputCPUImageType>;
  using ChangeInformationPointerType = typename ChangeInformationFilterType::Pointer;
  using RegionOfInterestFilterType = itk::RegionOfInterestImageFilter<OuputCPUImageType, OuputCPUImageType>;
  using RegionOfInterestPointerType = typename RegionOfInterestFilterType::Pointer;
  using MultiplyImageFilterType = itk::MultiplyImageFilter<OuputCPUImageType, OuputCPUImageType>;
  using MultpiplyImageFilterPointerType = typename MultiplyImageFilterType::Pointer;
  using ExpImageFilterType = itk::ExpImageFilter<OuputCPUImageType, OuputCPUImageType>;
  using ExpImageFilterPointerType = typename ExpImageFilterType::Pointer;

  /** ImageDimension constants */
  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;

  /** Typedef for the boundary condition */
  using BoundaryCondition = itk::ConstantBoundaryCondition<OuputCPUImageType>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ZengForwardProjectionImageFilter, ForwardProjectionImageFilter);

  /** Get / Set the sigma zero of the PSF. Default is 1.5417233052142099 */
  itkGetMacro(SigmaZero, float);
  itkSetMacro(SigmaZero, float);

  /** Get / Set the alpha of the PSF. Default is 0.016241189545787734 */
  itkGetMacro(Alpha, float);
  itkSetMacro(Alpha, float);

protected:
  ZengForwardProjectionImageFilter();
  virtual ~ZengForwardProjectionImageFilter() ITK_OVERRIDE {}

  void
  GenerateInputRequestedRegion() ITK_OVERRIDE;

  void
  GenerateOutputInformation() ITK_OVERRIDE;

  void
  GenerateData() ITK_OVERRIDE;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void
  VerifyInputInformation() const ITK_OVERRIDE;

  RegionOfInterestPointerType        m_RegionOfInterest;
  AddImageFilterPointerType          m_AddImageFilter;
  PasteImageFilterPointerType        m_PasteImageFilter;
  DiscreteGaussianFilterPointeurType m_DiscreteGaussianFilter;
  ResampleImageFilterPointerType     m_ResampleImageFilter;
  TransformPointerType               m_Transform;
  ChangeInformationPointerType       m_ChangeInformation;
  MultpiplyImageFilterPointerType    m_MultiplyImageFilter;
  MultpiplyImageFilterPointerType    m_AttenuationMapMultiplyImageFilter;
  RegionOfInterestPointerType        m_AttenuationMapRegionOfInterest;
  ExpImageFilterPointerType          m_AttenuationMapExpImageFilter;
  ResampleImageFilterPointerType     m_AttenuationMapResampleImageFilter;
  MultpiplyImageFilterPointerType    m_AttenuationMapConstantMultiplyImageFilter;
  ChangeInformationPointerType       m_AttenuationMapChangeInformation;
  BoundaryCondition                  m_BoundsCondition;


private:
  ZengForwardProjectionImageFilter(const Self &); // purposely not implemented
  void
  operator=(const Self &); // purposely not implemented

  float      m_SigmaZero;
  float      m_Alpha;
  VectorType m_VectorOrthogonalDetector;
  PointType  m_centerVolume;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkZengForwardProjectionImageFilter.hxx"
#endif

#endif
