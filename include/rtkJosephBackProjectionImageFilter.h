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

  /** \brief Function to multiply the interpolation weights with the sampled
   *         volume values.
   */
  using InterpolationWeightMultiplicationFunc =
    std::function<OutputPixelType(const ThreadIdType, double, const WeightCoordinateType, const InputPixelType *, int)>;

  /** \brief Function to compute the value to backproject from a projection
   *         sample.
   */
  using SumAlongRayFunc =
    std::function<OutputPixelType(const InputPixelType &, const InputPixelType, const VectorType &, bool &)>;

  /** \brief Function to apply the splat weight multiplication and accumulate
   *         the contribution into the output voxel.
   */
  using SplatWeightMultiplicationFunc = std::function<
    void(const InputPixelType &, OutputPixelType &, const double, const double, const WeightCoordinateType)>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(JosephBackProjectionImageFilter);

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

  itkGetMacro(InferiorClip, double);
  itkSetMacro(InferiorClip, double);
  itkGetMacro(SuperiorClip, double);
  itkSetMacro(SuperiorClip, double);

protected:
  JosephBackProjectionImageFilter();
  ~JosephBackProjectionImageFilter() override = default;

  void
  GenerateData() override;

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
                const CoordinateType   x,
                const CoordinateType   y,
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
                         const CoordinateType   x,
                         const CoordinateType   y,
                         const int              ox,
                         const int              oy,
                         const CoordinateType   minx,
                         const CoordinateType   miny,
                         const CoordinateType   maxx,
                         const CoordinateType   maxy);

  inline OutputPixelType
  BilinearInterpolation(const double           stepLengthInVoxel,
                        const InputPixelType * pxiyi,
                        const InputPixelType * pxsyi,
                        const InputPixelType * pxiys,
                        const InputPixelType * pxsys,
                        const CoordinateType   x,
                        const CoordinateType   y,
                        const int              ox,
                        const int              oy);

  inline OutputPixelType
  BilinearInterpolationOnBorders(const double           stepLengthInVoxel,
                                 const InputPixelType * pxiyi,
                                 const InputPixelType * pxsyi,
                                 const InputPixelType * pxiys,
                                 const InputPixelType * pxsys,
                                 const CoordinateType   x,
                                 const CoordinateType   y,
                                 const int              ox,
                                 const int              oy,
                                 const CoordinateType   minx,
                                 const CoordinateType   miny,
                                 const CoordinateType   maxx,
                                 const CoordinateType   maxy);

  // lambdas or std::functions
  // MUST BE initiated in constructor, or custom functions
  // must be set by means of Set methods in application
  SplatWeightMultiplicationFunc         m_SplatWeightMultiplication;
  InterpolationWeightMultiplicationFunc m_InterpolationWeightMultiplication;
  SumAlongRayFunc                       m_SumAlongRay;
  double                                m_InferiorClip{ 0. };
  double                                m_SuperiorClip{ 1. };
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkJosephBackProjectionImageFilter.hxx"
#endif

#endif
