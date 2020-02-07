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

#ifndef rtkJosephBackProjectionImageFilter_h
#define rtkJosephBackProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"

#include <itkVectorImage.h>

namespace rtk
{
namespace Functor
{
/** \class InterpolationWeightMultiplicationBackProjection
 * \brief Function to multiply the interpolation weights with the projected
 * volume values.
 *
 * \author Simon Rit
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TCoordRepType, class TOutput = TInput>
class InterpolationWeightMultiplicationBackProjection
{
public:
  InterpolationWeightMultiplicationBackProjection() = default;
  ~InterpolationWeightMultiplicationBackProjection() = default;
  bool
  operator!=(const InterpolationWeightMultiplicationBackProjection &) const
  {
    return false;
  }
  bool
  operator==(const InterpolationWeightMultiplicationBackProjection & other) const
  {
    return !(*this != other);
  }

  inline int
  operator()(const double        itkNotUsed(stepLengthInVoxel),
             const TCoordRepType itkNotUsed(weight),
             const TInput *      itkNotUsed(p),
             const int           itkNotUsed(i)) const
  {
    return 0;
  }
};

/** \class SumAlongRay
 * \brief Function to compute the attenuation correction on the projection.
 *
 * \author Antoine Robert
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TOutput>
class ValueAlongRay
{
public:
  using VectorType = itk::Vector<double, 3>;

  ValueAlongRay() = default;
  ~ValueAlongRay() = default;
  bool
  operator!=(const ValueAlongRay &) const
  {
    return false;
  }
  bool
  operator==(const ValueAlongRay & other) const
  {
    return !(*this != other);
  }

  inline const TOutput &
  operator()(const TInput &     rayValue,
             const TInput       itkNotUsed(attenuationRay),
             const VectorType & itkNotUsed(stepInMM),
             bool               itkNotUsed(isEndRay)) const
  {
    return rayValue;
  }
};
/** \class SplatWeightMultiplication
 * \brief Function to multiply the interpolation weights with the projection
 * values.
 *
 * \author Cyril Mory
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TCoordRepType, class TOutput = TCoordRepType>
class SplatWeightMultiplication
{
public:
  SplatWeightMultiplication() = default;
  ~SplatWeightMultiplication() = default;
  bool
  operator!=(const SplatWeightMultiplication &) const
  {
    return false;
  }
  bool
  operator==(const SplatWeightMultiplication & other) const
  {
    return !(*this != other);
  }

  inline void
  operator()(const TInput &      rayValue,
             TOutput &           output,
             const double        stepLengthInVoxel,
             const double        voxelSize,
             const TCoordRepType weight) const
  {
    output += rayValue * weight * voxelSize * stepLengthInVoxel;
  }
};

} // end namespace Functor


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

template <
  class TInputImage,
  class TOutputImage,
  class TInterpolationWeightMultiplication = Functor::InterpolationWeightMultiplicationBackProjection<
    typename TInputImage::PixelType,
    typename itk::PixelTraits<typename TInputImage::PixelType>::ValueType>,
  class TSplatWeightMultiplication =
    Functor::SplatWeightMultiplication<typename TInputImage::PixelType, double, typename TOutputImage::PixelType>,
  class TSumAlongRay = Functor::ValueAlongRay<typename TInputImage::PixelType, typename TOutputImage::PixelType>>
class ITK_EXPORT JosephBackProjectionImageFilter : public BackProjectionImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(JosephBackProjectionImageFilter);

  /** Standard class type alias. */
  using Self = JosephBackProjectionImageFilter;
  using Superclass = BackProjectionImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using CoordRepType = double;
  using VectorType = itk::Vector<CoordRepType, TInputImage::ImageDimension>;
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  using GeometryPointer = typename GeometryType::Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(JosephBackProjectionImageFilter, BackProjectionImageFilter);

  /** Get/Set the functor that is used to multiply each interpolation value with a volume value */
  TInterpolationWeightMultiplication &
  GetInterpolationWeightMultiplication()
  {
    return m_InterpolationWeightMultiplication;
  }
  const TInterpolationWeightMultiplication &
  GetInterpolationWeightMultiplication() const
  {
    return m_InterpolationWeightMultiplication;
  }
  void
  SetInterpolationWeightMultiplication(const TInterpolationWeightMultiplication & _arg)
  {
    if (m_InterpolationWeightMultiplication != _arg)
    {
      m_InterpolationWeightMultiplication = _arg;
      this->Modified();
    }
  }

  /** Get/Set the functor that is used to multiply each interpolation value with a volume value */
  TSplatWeightMultiplication &
  GetSplatWeightMultiplication()
  {
    return m_SplatWeightMultiplication;
  }
  const TSplatWeightMultiplication &
  GetSplatWeightMultiplication() const
  {
    return m_SplatWeightMultiplication;
  }
  void
  SetSplatWeightMultiplication(const TSplatWeightMultiplication & _arg)
  {
    if (m_SplatWeightMultiplication != _arg)
    {
      m_SplatWeightMultiplication = _arg;
      this->Modified();
    }
  }

  /** Get/Set the functor that is used to compute the sum along the ray*/
  TSumAlongRay &
  GetSumAlongRay()
  {
    return m_SumAlongRay;
  }
  const TSumAlongRay &
  GetSumAlongRay() const
  {
    return m_SumAlongRay;
  }
  void
  SetSumAlongRay(const TSumAlongRay & _arg)
  {
    if (m_SumAlongRay != _arg)
    {
      m_SumAlongRay = _arg;
      this->Modified();
    }
  }

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

  inline void
  BilinearSplat(const InputPixelType & rayValue,
                const double           stepLengthInVoxel,
                const double           voxelSize,
                OutputPixelType *      pxiyi,
                OutputPixelType *      pxsyi,
                OutputPixelType *      pxiys,
                OutputPixelType *      pxsys,
                const double           x,
                const double           y,
                const int              ox,
                const int              oy);

  inline void
  BilinearSplatOnBorders(const InputPixelType & rayValue,
                         const double           stepLengthInVoxel,
                         const double           voxelSize,
                         OutputPixelType *      pxiyi,
                         OutputPixelType *      pxsyi,
                         OutputPixelType *      pxiys,
                         OutputPixelType *      pxsys,
                         const double           x,
                         const double           y,
                         const int              ox,
                         const int              oy,
                         const CoordRepType     minx,
                         const CoordRepType     miny,
                         const CoordRepType     maxx,
                         const CoordRepType     maxy);

  inline OutputPixelType
  BilinearInterpolation(const double           stepLengthInVoxel,
                        const InputPixelType * pxiyi,
                        const InputPixelType * pxsyi,
                        const InputPixelType * pxiys,
                        const InputPixelType * pxsys,
                        const double           x,
                        const double           y,
                        const int              ox,
                        const int              oy);

  inline OutputPixelType
  BilinearInterpolationOnBorders(const double           stepLengthInVoxel,
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

  /** Functor */
  TSplatWeightMultiplication         m_SplatWeightMultiplication;
  TInterpolationWeightMultiplication m_InterpolationWeightMultiplication;
  TSumAlongRay                       m_SumAlongRay;
  double                             m_InferiorClip{ 0. };
  double                             m_SuperiorClip{ 1. };
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkJosephBackProjectionImageFilter.hxx"
#endif

#endif
