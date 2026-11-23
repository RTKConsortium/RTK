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
#include <itkPixelTraits.h>

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
  /** Function to multiply the interpolation weights with the sampled volume
   * values. */
  using InterpolationWeightMultiplicationFunc =
    std::function<OutputPixelType(double, const WeightCoordinateType, const InputPixelType *, ptrdiff_t)>;

  /** Function to compute the value to backproject from a projection sample.
   * It returns a reference which must remain valid until the sample has been
   * splat, e.g. a reference to the input projection pixel or to a data member.
   * Lambdas must therefore explicitly declare their return type as
   * const OutputPixelType &, otherwise they return by value and the back
   * projection silently reads a dangling reference. */
  using SumAlongRayFunc =
    std::function<const OutputPixelType &(const InputPixelType &, const InputPixelType, const VectorType &, bool &)>;

  /** Function to apply the splat weight multiplication and accumulate the
   * contribution into the output voxel. */
  using SplatWeightMultiplicationFunc =
    std::function<void(const InputPixelType &, OutputPixelType &, const double, const double, const double)>;

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

  /** Get/Set the lambda function that is used to multiply each splat weight with a projection value */
  SplatWeightMultiplicationFunc &
  GetSplatWeightMultiplication()
  {
    return m_SplatWeightMultiplication;
  }
  const SplatWeightMultiplicationFunc &
  GetSplatWeightMultiplication() const
  {
    return m_SplatWeightMultiplication;
  }
  void
  SetSplatWeightMultiplication(const SplatWeightMultiplicationFunc & _arg)
  {
    m_SplatWeightMultiplication = _arg;
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
  /** Back projects the projections with the given functions. It is templated
   * over their types so that they can be inlined in the ray stepping loop when
   * they are lambdas rather than std::function. */
  template <class TInterpolationWeightMultiplication, class TSplatWeightMultiplication, class TSumAlongRay>
  void
  BackProject(TInterpolationWeightMultiplication & interpolationWeightMultiplication,
              TSplatWeightMultiplication &         splatWeightMultiplication,
              TSumAlongRay &                       sumAlongRay);

  /** Calls BackProject with the lambdas stored in the std::function members if
   * they are of the given types and returns true, returns false otherwise.
   * Subclasses which set their own lambdas call it in GenerateData with the
   * types of these lambdas to avoid indirect calls at each step. */
  template <class TInterpolationWeightMultiplication, class TSplatWeightMultiplication, class TSumAlongRay>
  bool
  BackProjectIfFunctionTypesAre();

  /** Default lambdas. Their types are used to detect that they have not been
   * replaced. */
  static auto
  MakeDefaultInterpolationWeightMultiplication()
  {
    // The interpolated value is only used by SumAlongRay for attenuation correction, which the default
    // SumAlongRay ignores, so do not read the volume
    return [](double                     itkNotUsed(stepLengthInVoxel),
              const WeightCoordinateType itkNotUsed(weight),
              const InputPixelType *     itkNotUsed(p),
              ptrdiff_t                  itkNotUsed(i)) -> OutputPixelType { return {}; };
  }
  static auto
  MakeDefaultSplatWeightMultiplication()
  {
    return [](const InputPixelType & rayValue,
              OutputPixelType &      output,
              const double           stepLengthInVoxel,
              const double           voxelSize,
              const double           weight) {
      output += static_cast<OutputPixelType>(rayValue * weight * voxelSize * stepLengthInVoxel);
    };
  }
  static auto
  MakeDefaultSumAlongRay()
  {
    return [](const InputPixelType & rayValue,
              const InputPixelType   itkNotUsed(attenuationRay),
              const VectorType &     itkNotUsed(stepInMM),
              bool &                 itkNotUsed(isEndRay)) -> const OutputPixelType & { return rayValue; };
  }
#endif

  template <class TSplatWeightMultiplication>
  inline void
  BilinearSplat(TSplatWeightMultiplication & splatWeightMultiplication,
                const InputPixelType &       rayValue,
                double                       stepLengthInVoxel,
                double                       voxelSize,
                OutputPixelType *            pxiyi,
                OutputPixelType *            pxsyi,
                OutputPixelType *            pxiys,
                OutputPixelType *            pxsys,
                CoordinateType               x,
                CoordinateType               y,
                ptrdiff_t                    ox,
                ptrdiff_t                    oy);

  template <class TSplatWeightMultiplication>
  inline void
  BilinearSplatOnBorders(TSplatWeightMultiplication & splatWeightMultiplication,
                         const InputPixelType &       rayValue,
                         double                       stepLengthInVoxel,
                         double                       voxelSize,
                         OutputPixelType *            pxiyi,
                         OutputPixelType *            pxsyi,
                         OutputPixelType *            pxiys,
                         OutputPixelType *            pxsys,
                         CoordinateType               x,
                         CoordinateType               y,
                         ptrdiff_t                    ox,
                         ptrdiff_t                    oy,
                         CoordinateType               minx,
                         CoordinateType               miny,
                         CoordinateType               maxx,
                         CoordinateType               maxy);

  template <class TInterpolationWeightMultiplication>
  inline OutputPixelType
  BilinearInterpolation(TInterpolationWeightMultiplication & interpolationWeightMultiplication,
                        double                               stepLengthInVoxel,
                        const InputPixelType *               pxiyi,
                        const InputPixelType *               pxsyi,
                        const InputPixelType *               pxiys,
                        const InputPixelType *               pxsys,
                        CoordinateType                       x,
                        CoordinateType                       y,
                        ptrdiff_t                            ox,
                        ptrdiff_t                            oy);

  template <class TInterpolationWeightMultiplication>
  inline OutputPixelType
  BilinearInterpolationOnBorders(TInterpolationWeightMultiplication & interpolationWeightMultiplication,
                                 double                               stepLengthInVoxel,
                                 const InputPixelType *               pxiyi,
                                 const InputPixelType *               pxsyi,
                                 const InputPixelType *               pxiys,
                                 const InputPixelType *               pxsys,
                                 CoordinateType                       x,
                                 CoordinateType                       y,
                                 ptrdiff_t                            ox,
                                 ptrdiff_t                            oy,
                                 CoordinateType                       minx,
                                 CoordinateType                       miny,
                                 CoordinateType                       maxx,
                                 CoordinateType                       maxy);

#if !defined(ITK_WRAPPING_PARSER)
  // lambdas or std::functions
  // MUST BE initiated in constructor, or custom functions
  // must be set by means of Set methods in application
  SplatWeightMultiplicationFunc         m_SplatWeightMultiplication;
  InterpolationWeightMultiplicationFunc m_InterpolationWeightMultiplication;
  SumAlongRayFunc                       m_SumAlongRay;
#endif
  double m_InferiorClip{ 0. };
  double m_SuperiorClip{ 1. };
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkJosephBackProjectionImageFilter.hxx"
#endif

#endif
