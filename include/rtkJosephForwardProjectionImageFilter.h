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

#ifndef rtkJosephForwardProjectionImageFilter_h
#define rtkJosephForwardProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkMacro.h"
#include <itkPixelTraits.h>

#include "rtkProjectionsRegionConstIteratorRayBased.h"

#include <itkVectorImage.h>
namespace rtk
{

/** \class JosephForwardProjectionImageFilter
 * \brief Joseph forward projection.
 *
 * Performs a forward projection, i.e. accumulation along x-ray lines,
 * using [Joseph, IEEE TMI, 1982]. The forward projector tests if the  detector
 * has been placed after the source and the volume. If the detector is in the volume
 * the ray tracing is performed only until that point.
 *
 * \test rtkforwardprojectiontest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK Projector
 */
template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT JosephForwardProjectionImageFilter
  : public ForwardProjectionImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(JosephForwardProjectionImageFilter);

  /** Standard class type alias. */
  using Self = JosephForwardProjectionImageFilter;
  using Superclass = ForwardProjectionImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using CoordinateType = double;
  using WeightCoordinateType = typename itk::PixelTraits<InputPixelType>::ValueType;
  using VectorType = itk::Vector<CoordinateType, TInputImage::ImageDimension>;
  using TClipImageType = itk::Image<double, TOutputImage::ImageDimension>;
  using TClipImagePointer = typename TClipImageType::Pointer;

  /** \brief Function to multiply the interpolation weights with the projected
   * volume values.
   *
   * \author Simon Rit
   */
  using InterpolationWeightMultiplicationFunc =
    std::function<OutputPixelType(const ThreadIdType, double, const WeightCoordinateType, const InputPixelType *, int)>;

  /** \brief Function to compute the attenuation correction on the projection.
   *
   * \author Antoine Robert
   */
  using SumAlongRayFunc =
    std::function<void(const ThreadIdType, OutputPixelType &, const InputPixelType, const VectorType &)>;

  /** \brief Function to accumulate the ray casting on the projection.
   *
   * \author Simon Rit
   */
  using ProjectedValueAccumulationFunc = std::function<void(const ThreadIdType,
                                                            const InputPixelType &,
                                                            OutputPixelType &,
                                                            const OutputPixelType &,
                                                            const VectorType &,
                                                            const VectorType &,
                                                            const VectorType &,
                                                            const VectorType &,
                                                            const VectorType &)>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(JosephForwardProjectionImageFilter);

  /** Get/Set the lambda function that is used to multiply each interpolation value with a volume value */
  InterpolationWeightMultiplicationFunc &
  GetInterpolationWeightMultiplication()
  {
    return m_InterpolationWeightMultiplication;
  }
  const InterpolationWeightMultiplicationFunc &
  GetInterpolationWeightMultiplication() const
  {
    return m_InterpolationWeightMultiplication;
  }
  void
  SetInterpolationWeightMultiplication(const InterpolationWeightMultiplicationFunc & _arg)
  {
    m_InterpolationWeightMultiplication = _arg;
    this->Modified();
  }

  /** Get/Set the lambda function that is used to accumulate values in the projection image after the ray
   * casting has been performed. */
  ProjectedValueAccumulationFunc &
  GetProjectedValueAccumulation()
  {
    return m_ProjectedValueAccumulation;
  }
  const ProjectedValueAccumulationFunc &
  GetProjectedValueAccumulation() const
  {
    return m_ProjectedValueAccumulation;
  }
  void
  SetProjectedValueAccumulation(const ProjectedValueAccumulationFunc & _arg)
  {
    m_ProjectedValueAccumulation = _arg;
    this->Modified();
  }

  /** Get/Set the lambda function that is used to compute the sum along the ray */
  SumAlongRayFunc &
  GetSumAlongRay()
  {
    return m_SumAlongRay;
  }
  const SumAlongRayFunc &
  GetSumAlongRay() const
  {
    return m_SumAlongRay;
  }
  void
  SetSumAlongRay(const SumAlongRayFunc & _arg)
  {
    m_SumAlongRay = _arg;
    this->Modified();
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

  /** Each ray is clipped from source+m_InferiorClip*(pixel-source) to
  ** source+m_SuperiorClip*(pixel-source) with m_InferiorClip and
  ** m_SuperiorClip equal 0 and 1 by default. */
  itkGetMacro(InferiorClip, double);
  itkSetMacro(InferiorClip, double);
  itkGetMacro(SuperiorClip, double);
  itkSetMacro(SuperiorClip, double);

protected:
  JosephForwardProjectionImageFilter();
  ~JosephForwardProjectionImageFilter() override = default;

  /** Apply changes to the input image requested region. */
  void
  GenerateInputRequestedRegion() override;

  void
  ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId) override;

  /** If a third input is given, it should be in the same physical space
   * as the first one. */
  void
  VerifyInputInformation() const override;

  inline OutputPixelType
  BilinearInterpolation(const ThreadIdType     threadId,
                        const double           stepLengthInVoxel,
                        const InputPixelType * pxiyi,
                        const InputPixelType * pxsyi,
                        const InputPixelType * pxiys,
                        const InputPixelType * pxsys,
                        const double           x,
                        const double           y,
                        const int              ox,
                        const int              oy);

  inline OutputPixelType
  BilinearInterpolationOnBorders(const ThreadIdType     threadId,
                                 const double           stepLengthInVoxel,
                                 const InputPixelType * pxiyi,
                                 const InputPixelType * pxsyi,
                                 const InputPixelType * pxiys,
                                 const InputPixelType * pxsys,
                                 const double           x,
                                 const double           y,
                                 const int              ox,
                                 const int              oy,
                                 const double           minx,
                                 const double           miny,
                                 const double           maxx,
                                 const double           maxy);

private:
  // lambdas or std::functions
  // MUST BE initiated in constructor, or custom functions
  // must be set by means of Set methods in application
  InterpolationWeightMultiplicationFunc m_InterpolationWeightMultiplication;
  SumAlongRayFunc                       m_SumAlongRay;
  ProjectedValueAccumulationFunc        m_ProjectedValueAccumulation;
  double                                m_InferiorClip{ 0. };
  double                                m_SuperiorClip{ 1. };
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkJosephForwardProjectionImageFilter.hxx"
#endif

#endif
