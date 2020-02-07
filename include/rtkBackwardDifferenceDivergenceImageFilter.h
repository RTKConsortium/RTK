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

#ifndef rtkBackwardDifferenceDivergenceImageFilter_h
#define rtkBackwardDifferenceDivergenceImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkCastImageFilter.h>

namespace rtk
{
/** \class BackwardDifferenceDivergenceImageFilter
 * \brief Computes the backward differences divergence
 * (adjoint of the forward differences gradient) of the input image
 *
 * The exact definition of the desired divergence filter can
 * be found in Chambolle, Antonin. "An Algorithm for Total
 * Variation Minimization and Applications." J. Math. Imaging Vis. 20,
 * no. 1-2 (January 2004): 89-97.
 *
 * \ingroup RTK IntensityImageFilters
 */

template <typename TInputImage, typename TOutputImage = itk::Image<float, TInputImage::ImageDimension>>
class BackwardDifferenceDivergenceImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(BackwardDifferenceDivergenceImageFilter);

  /** Extract dimension from input and output image. */
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);

  /** Convenient type alias for simplifying declarations. */
  using InputImageType = TInputImage;

  /** Standard class type alias. */
  using Self = BackwardDifferenceDivergenceImageFilter;
  using Superclass = itk::ImageToImageFilter<InputImageType, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BackwardDifferenceDivergenceImageFilter, ImageToImageFilter);

  /** Use the image spacing information in calculations. Use this option if you
   *  want derivatives in physical space. Default is UseImageSpacingOn. */
  void
  SetUseImageSpacingOn()
  {
    this->SetUseImageSpacing(true);
  }

  /** Ignore the image spacing. Use this option if you want derivatives in
      isotropic pixel space.  Default is UseImageSpacingOn. */
  void
  SetUseImageSpacingOff()
  {
    this->SetUseImageSpacing(false);
  }

  /** Set/Get whether or not the filter will use the spacing of the input
      image in its calculations */
  itkSetMacro(UseImageSpacing, bool);
  itkGetConstMacro(UseImageSpacing, bool);

  /** Set along which dimensions the gradient computation should be
      performed. The vector components at unprocessed dimensions are ignored */
  void
  SetDimensionsProcessed(bool * DimensionsProcessed);

  /** Allows to change the default boundary condition */
  void
  OverrideBoundaryCondition(itk::ImageBoundaryCondition<TInputImage> * boundaryCondition);

  /** Image type alias support. */
  using InputPixelType = typename InputImageType::PixelType;
  using InputImageRegionType = typename InputImageType::RegionType;
  using InputSizeType = typename InputImageType::SizeType;
  using CovariantVectorType = itk::CovariantVector<InputPixelType, InputImageDimension>;

protected:
  BackwardDifferenceDivergenceImageFilter();
  ~BackwardDifferenceDivergenceImageFilter() override;

  void
  GenerateInputRequestedRegion() override;

  void
  BeforeThreadedGenerateData() override;

  void
  DynamicThreadedGenerateData(const typename InputImageType::RegionType & outputRegionForThread) override;

  void
  AfterThreadedGenerateData() override;

private:
  bool                              m_UseImageSpacing;
  typename TInputImage::SpacingType m_InvSpacingCoeffs;

  // list of the dimensions along which the divergence has
  // to be computed. The components on other dimensions
  // are ignored for performance, but the gradient filter
  // sets them to zero anyway
  bool m_DimensionsProcessed[TInputImage::ImageDimension];

  // The default is ConstantBoundaryCondition, but this behavior sometimes needs to be overriden
  itk::ImageBoundaryCondition<TInputImage, TInputImage> * m_BoundaryCondition;
  // If so, do not perform boundary processing in AfterThreadedGenerateData
  bool m_IsBoundaryConditionOverriden;
};

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkBackwardDifferenceDivergenceImageFilter.hxx"
#endif

#endif //__rtkBackwardDifferenceDivergenceImageFilter__
