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
#include <itkMath.h>
#include <itkPixelTraits.h>

#include "rtkProjectionsRegionConstIteratorRayBased.h"

#include <itkVectorImage.h>

#include <array>
#include <functional>
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
  using PointType = typename TInputImage::PointType;
  using CoordinateType = double;
  using WeightCoordinateType = typename itk::PixelTraits<InputPixelType>::ValueType;
  using VectorType = itk::Vector<CoordinateType, TInputImage::ImageDimension>;
  using TClipImageType = itk::Image<double, TOutputImage::ImageDimension>;
  using TClipImagePointer = typename TClipImageType::Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(JosephForwardProjectionImageFilter);

#if !defined(ITK_WRAPPING_PARSER)
  /** Bilinear weights and offsets of the four corners of a ray step. The
   * offsets are relative to the start of the volume buffer. */
  using InterpolationWeightsType = std::array<WeightCoordinateType, 4>;
  using CornerOffsetsType = std::array<ptrdiff_t, 4>;

  /** Bilinear weights, volume values and offsets of the four corners of a ray step. */
  struct StepSamples
  {
    InterpolationWeightsType      weights;
    std::array<InputPixelType, 4> values;
    CornerOffsetsType             offsets;
    /** Length of the step along the main direction in voxels, smaller than 1 at the ends of the ray. */
    double stepLengthInVoxel;
    /** Length in mm of a step of one voxel along the main direction. */
    double stepSizeInMM;

    /** Volume interpolated at the step, multiplied by stepLengthInVoxel. */
    OutputPixelType
    Interpolate() const
    {
      OutputPixelType result = weights[0] * values[0];
      for (unsigned int k = 1; k < 4; ++k)
      {
        result += weights[k] * values[k];
      }
      return static_cast<OutputPixelType>(result * stepLengthInVoxel);
    }
  };

  /** Per-ray state of the step functions which need none. */
  struct EmptyRayState
  {};

  /** Function to accumulate the value computed along a ray in the projection. */
  using ProjectedValueAccumulationFunc = std::function<void(const InputPixelType &,
                                                            OutputPixelType &,
                                                            const OutputPixelType &,
                                                            const VectorType &,
                                                            const PointType &,
                                                            const VectorType &,
                                                            const PointType &,
                                                            const PointType &)>;

  /** Set the function called at each ray step as step(rayValue, state, samples).
   * rayValue is then passed to ProjectedValueAccumulation and state is a copy of
   * initialState for each ray. The function is called concurrently for different
   * rays and inlined in the ray casting loop. */
  template <class TStep, class TRayState = EmptyRayState>
  void
  SetStepFunction(TStep step, TRayState initialState = TRayState())
  {
    m_RayCast = [step = std::move(step), initialState](Self & filter, const OutputImageRegionType & region) {
      filter.RayCast(region, step, initialState);
    };
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
#endif

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

#if !defined(ITK_WRAPPING_PARSER)
  /** Bilinear weights and buffer offsets of the four corners of a ray step. */
  template <class TWeight>
  struct RayStencil
  {
    std::array<TWeight, 4> weights;
    CornerOffsetsType      offsets;
  };

  /** Stencil at (x, y) in the slice starting at offset slice of the buffer.
   * On borders, corners outside [minx, maxx] x [miny, maxy] are replaced by
   * their neighbor inside, possibly repeating an offset. */
  template <bool VOnBorder, class TWeight>
  static RayStencil<TWeight>
  MakeRayStencil(ptrdiff_t slice,
                 double    x,
                 double    y,
                 ptrdiff_t ox,
                 ptrdiff_t oy,
                 double    minx = 0.,
                 double    miny = 0.,
                 double    maxx = 0.,
                 double    maxy = 0.)
  {
    const ptrdiff_t ix = itk::Math::floor(x);
    const ptrdiff_t iy = itk::Math::floor(y);
    const ptrdiff_t idx = slice + ix * ox + iy * oy;
    const double    lx = x - ix;
    const double    ly = y - iy;
    const double    lxc = 1. - lx;
    const double    lyc = 1. - ly;

    if constexpr (VOnBorder)
    {
      const ptrdiff_t xi = ix < minx ? ox : 0;
      const ptrdiff_t yi = iy < miny ? oy : 0;
      const ptrdiff_t xs = ix >= maxx ? 0 : ox;
      const ptrdiff_t ys = iy >= maxy ? 0 : oy;
      return { { static_cast<TWeight>(lxc * lyc),
                 static_cast<TWeight>(lxc * ly),
                 static_cast<TWeight>(lx * lyc),
                 static_cast<TWeight>(lx * ly) },
               { idx + xi + yi, idx + xi + ys, idx + xs + yi, idx + xs + ys } };
    }
    else
    {
      return { { static_cast<TWeight>(lxc * lyc),
                 static_cast<TWeight>(lx * lyc),
                 static_cast<TWeight>(lxc * ly),
                 static_cast<TWeight>(lx * ly) },
               { idx, idx + ox, idx + oy, idx + ox + oy } };
    }
  }

  /** Samples of the ray step at (x, y) in the slice starting at offset slice of buffer. */
  template <bool VOnBorder>
  static StepSamples
  MakeStepSamples(const InputPixelType * buffer,
                  ptrdiff_t              slice,
                  double                 x,
                  double                 y,
                  ptrdiff_t              ox,
                  ptrdiff_t              oy,
                  double                 stepLengthInVoxel,
                  double                 stepSizeInMM,
                  double                 minx = 0.,
                  double                 miny = 0.,
                  double                 maxx = 0.,
                  double                 maxy = 0.)
  {
    const auto   stencil = MakeRayStencil<VOnBorder, WeightCoordinateType>(slice, x, y, ox, oy, minx, miny, maxx, maxy);
    const auto & o = stencil.offsets;
    return {
      stencil.weights, { buffer[o[0]], buffer[o[1]], buffer[o[2]], buffer[o[3]] }, o, stepLengthInVoxel, stepSizeInMM
    };
  }
#endif

private:
#if !defined(ITK_WRAPPING_PARSER)
  /** Ray casts the projection pixels of outputRegionForThread with step, which
   * is inlined in the ray stepping loop. */
  template <class TStep, class TRayState>
  void
  RayCast(const OutputImageRegionType & outputRegionForThread, const TStep & step, const TRayState & initialState);

  std::function<void(Self &, const OutputImageRegionType &)> m_RayCast;
  ProjectedValueAccumulationFunc                             m_ProjectedValueAccumulation;
#endif
  double m_InferiorClip{ 0. };
  double m_SuperiorClip{ 1. };
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkJosephForwardProjectionImageFilter.hxx"
#endif

#endif
