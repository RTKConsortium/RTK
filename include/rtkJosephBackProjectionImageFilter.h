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

#ifndef rtkJosephBackProjectionImageFilter_h
#define rtkJosephBackProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"

#include <itkVectorImage.h>
#include <itkMath.h>
#include <itkPixelTraits.h>

#include <array>
#include <functional>

namespace rtk
{
/** \class JosephBackProjectionImageFilter
 * \brief Joseph back projection.
 *
 * Performs a back projection, i.e. smearing of ray value along its path,
 * using [Joseph, IEEE TMI, 1982]. The back projector is the adjoint operator of the
 * forward projector
 *
 * \test rtkbackprojectiontest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK Projector
 */
template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT JosephBackProjectionImageFilter : public BackProjectionImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(JosephBackProjectionImageFilter);

  /** Standard class type alias. */
  using Self = JosephBackProjectionImageFilter;
  using Superclass = BackProjectionImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;
  using WeightCoordinateType = typename itk::PixelTraits<InputPixelType>::ValueType;

  using OutputImageRegionType = typename TOutputImage::RegionType;
  using CoordinateType = double;
  using VectorType = itk::Vector<CoordinateType, TInputImage::ImageDimension>;
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  using GeometryPointer = typename GeometryType::Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(JosephBackProjectionImageFilter);

#if !defined(ITK_WRAPPING_PARSER)
  /** Bilinear weights and offsets of the four corners of a ray step. The
   * offsets are relative to the start of the volume buffer. */
  using SplatWeightsType = std::array<double, 4>;
  using CornerOffsetsType = std::array<ptrdiff_t, 4>;

  /** Projection value, bilinear weights and volume offsets of the four corners of a ray step. */
  struct StepSamples
  {
    InputPixelType projectionValue;
    /** Offset of the projection pixel from the start of the projection buffer. */
    size_t            projectionOffset;
    SplatWeightsType  weights;
    CornerOffsetsType offsets;
    /** Length of the step along the main direction in voxels, smaller than 1 at the ends of the ray. */
    double stepLengthInVoxel;
    /** Length in mm of a step of one voxel along the main direction. */
    double stepSizeInMM;
  };

  /** Per-ray state of the step functions which need none. */
  struct EmptyRayState
  {};

  /** Set the function called at each ray step as step(volume, state, samples).
   * volume points at the start of the output buffer, in which the function
   * accumulates the contribution of the step, and state is a copy of
   * initialState for each ray. The function is inlined in the ray stepping loop. */
  template <class TStep, class TRayState = EmptyRayState>
  void
  SetStepFunction(TStep step, TRayState initialState = TRayState())
  {
    m_BackProject = [step = std::move(step), initialState](Self & filter) { filter.BackProject(step, initialState); };
    this->Modified();
  }
#endif

  /** Each ray is clipped from source+m_InferiorClip*(pixel-source) to
  ** source+m_SuperiorClip*(pixel-source) with m_InferiorClip and
  ** m_SuperiorClip equal 0 and 1 by default. */
  itkGetMacro(InferiorClip, double);
  itkSetMacro(InferiorClip, double);
  itkGetMacro(SuperiorClip, double);
  itkSetMacro(SuperiorClip, double);

protected:
  JosephBackProjectionImageFilter();
  ~JosephBackProjectionImageFilter() override = default;

  void
  GenerateData() override;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void
  VerifyInputInformation() const override
  {}

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

  /** Samples of the ray step at (x, y) in the slice starting at offset slice of the volume buffer. */
  template <bool VOnBorder>
  static StepSamples
  MakeStepSamples(const InputPixelType & projectionValue,
                  size_t                 projectionOffset,
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
    const auto stencil = MakeRayStencil<VOnBorder, double>(slice, x, y, ox, oy, minx, miny, maxx, maxy);
    return { projectionValue, projectionOffset, stencil.weights, stencil.offsets, stepLengthInVoxel, stepSizeInMM };
  }
#endif

  double m_InferiorClip{ 0. };
  double m_SuperiorClip{ 1. };

private:
#if !defined(ITK_WRAPPING_PARSER)
  /** Back projects the projections with step, which is inlined in the ray stepping loop. */
  template <class TStep, class TRayState>
  void
  BackProject(const TStep & step, const TRayState & initialState);

  std::function<void(Self &)> m_BackProject;
#endif
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkJosephBackProjectionImageFilter.hxx"
#endif

#endif
