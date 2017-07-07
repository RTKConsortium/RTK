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

#ifndef rtkJosephForwardProjectionImageFilter_h
#define rtkJosephForwardProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkMacro.h"

#include "rtkRayBoxIntersectionFunction.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"

#include <itkVectorImage.h>
namespace rtk
{
namespace Functor
{
/** \class InterpolationWeightMultiplication
 * \brief Function to multiply the interpolation weights with the projected
 * volume values.
 *
 * \author Simon Rit
 *
 * \ingroup Functions
 */
template< class TInput, class TCoordRepType, class TOutput=TCoordRepType >
class InterpolationWeightMultiplication
{
public:
  InterpolationWeightMultiplication() {};
  ~InterpolationWeightMultiplication() {};
  bool operator!=( const InterpolationWeightMultiplication & ) const {
    return false;
  }
  bool operator==(const InterpolationWeightMultiplication & other) const
  {
    return !( *this != other );
  }

  inline void operator()( const ThreadIdType itkNotUsed(threadId),
                             const double itkNotUsed(stepLengthInVoxel),
                             const TCoordRepType weight,
                             const TInput *p,
                             const int i,
                             const int itkNotUsed(vectorLength),
                             TOutput &result) const
  {  result += weight * p[i]; }
};

template< class TInput, class TCoordRepType, class TOutput=TCoordRepType >
class VectorInterpolationWeightMultiplication
{
public:
  VectorInterpolationWeightMultiplication() {};
  ~VectorInterpolationWeightMultiplication() {};
  bool operator!=( const VectorInterpolationWeightMultiplication & ) const {
    return false;
  }
  bool operator==(const VectorInterpolationWeightMultiplication & other) const
  {
    return !( *this != other );
  }

inline void operator()( const ThreadIdType itkNotUsed(threadId),
                           const double itkNotUsed(stepLengthInVoxel),
                           const TCoordRepType weight,
                           const TInput *p,
                           const int i,
                           const int vectorLength,
                           TOutput &result) const
  {
  for (int component=0; component<vectorLength; component++)
    result[component] += weight * p[i + component];
  }
};

/** \class ProjectedValueAccumulation
 * \brief Function to accumulate the ray casting on the projection.
 *
 * \author Simon Rit
 *
 * \ingroup Functions
 */
template< class TInput, class TOutput >
class VectorProjectedValueAccumulation
{
public:
  typedef itk::Vector<double, 3> VectorType;

  VectorProjectedValueAccumulation() {};
  ~VectorProjectedValueAccumulation() {};
  bool operator!=( const VectorProjectedValueAccumulation & ) const
    {
    return false;
    }
  bool operator==(const VectorProjectedValueAccumulation & other) const
    {
    return !( *this != other );
    }

  inline TOutput operator()(  const ThreadIdType itkNotUsed(threadId),
                              const TInput &input,
                              TOutput current,
                              const TOutput &rayCastValue,
                              const VectorType &stepInMM,
                              const VectorType &itkNotUsed(source),
                              const VectorType &itkNotUsed(sourceToPixel),
                              const VectorType &itkNotUsed(nearestPoint),
                              const VectorType &itkNotUsed(farthestPoint)) const
    {
    return (current + input + rayCastValue * stepInMM.GetNorm());
    }
};

template< class TInput, class TOutput >
class ProjectedValueAccumulation
{
public:
  typedef itk::Vector<double, 3> VectorType;

  ProjectedValueAccumulation() {};
  ~ProjectedValueAccumulation() {};
  bool operator!=( const ProjectedValueAccumulation & ) const
    {
    return false;
    }
  bool operator==(const ProjectedValueAccumulation & other) const
    {
    return !( *this != other );
    }

  inline void operator()( const ThreadIdType itkNotUsed(threadId),
                          const TInput &input,
                          TOutput &output,
                          const TOutput &rayCastValue,
                          const VectorType &stepInMM,
                          const VectorType &itkNotUsed(source),
                          const VectorType &itkNotUsed(sourceToPixel),
                          const VectorType &itkNotUsed(nearestPoint),
                          const VectorType &itkNotUsed(farthestPoint)) const
    {
    output = input + rayCastValue * stepInMM.GetNorm();
    }
};


} // end namespace Functor


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
 * \ingroup Projector
 */

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication = Functor::InterpolationWeightMultiplication<typename TInputImage::InternalPixelType, double, typename TOutputImage::PixelType>,
          class TProjectedValueAccumulation        = Functor::ProjectedValueAccumulation<typename TInputImage::PixelType, typename TOutputImage::PixelType> >
class ITK_EXPORT JosephForwardProjectionImageFilter :
  public ForwardProjectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef JosephForwardProjectionImageFilter                     Self;
  typedef ForwardProjectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                Pointer;
  typedef itk::SmartPointer<const Self>                          ConstPointer;
  typedef typename TInputImage::PixelType                        InputPixelType;
  typedef typename TInputImage::InternalPixelType                InputInternalPixelType;
  typedef typename TOutputImage::PixelType                       OutputPixelType;
  typedef typename TOutputImage::RegionType                      OutputImageRegionType;
  typedef typename TOutputImage::InternalPixelType               OutputInternalPixelType;
  typedef double                                                 CoordRepType;
  typedef itk::Vector<CoordRepType, TInputImage::ImageDimension> VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(JosephForwardProjectionImageFilter, ForwardProjectionImageFilter);

  /** Get/Set the functor that is used to multiply each interpolation value with a volume value */
  TInterpolationWeightMultiplication &       GetInterpolationWeightMultiplication() { return m_InterpolationWeightMultiplication; }
  const TInterpolationWeightMultiplication & GetInterpolationWeightMultiplication() const { return m_InterpolationWeightMultiplication; }
  void SetInterpolationWeightMultiplication(const TInterpolationWeightMultiplication & _arg)
    {
    if ( m_InterpolationWeightMultiplication != _arg )
      {
      m_InterpolationWeightMultiplication = _arg;
      this->Modified();
      }
    }

  /** Get/Set the functor that is used to accumulate values in the projection image after the ray
   * casting has been performed. */
  TProjectedValueAccumulation &       GetProjectedValueAccumulation() { return m_ProjectedValueAccumulation; }
  const TProjectedValueAccumulation & GetProjectedValueAccumulation() const { return m_ProjectedValueAccumulation; }
  void SetProjectedValueAccumulation(const TProjectedValueAccumulation & _arg)
    {
    if ( m_ProjectedValueAccumulation != _arg )
      {
      m_ProjectedValueAccumulation = _arg;
      this->Modified();
      }
    }

protected:
  JosephForwardProjectionImageFilter() {}
  ~JosephForwardProjectionImageFilter() {}

  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  inline void BilinearInterpolation(const ThreadIdType threadId,
                                               const double stepLengthInVoxel,
                                               const InputInternalPixelType *pxiyi,
                                               const InputInternalPixelType *pxsyi,
                                               const InputInternalPixelType *pxiys,
                                               const InputInternalPixelType *pxsys,
                                               const double x,
                                               const double y,
                                               const int ox,
                                               const int oy,
                                               OutputPixelType &sum);

  inline void BilinearInterpolationOnBorders(const ThreadIdType threadId,
                                               const double stepLengthInVoxel,
                                               const InputInternalPixelType *pxiyi,
                                               const InputInternalPixelType *pxsyi,
                                               const InputInternalPixelType *pxiys,
                                               const InputInternalPixelType *pxsys,
                                               const double x,
                                               const double y,
                                               const int ox,
                                               const int oy,
                                               const double minx,
                                               const double miny,
                                               const double maxx,
                                               const double maxy,
                                               OutputPixelType &sum);

  void Accumulate(ThreadIdType threadId,
                  rtk::ProjectionsRegionConstIteratorRayBased<TInputImage> *itIn,
                  itk::ImageRegionIteratorWithIndex<TOutputImage> itOut,
                  typename TOutputImage::PixelType sum,
                  typename rtk::RayBoxIntersectionFunction<CoordRepType, TOutputImage::ImageDimension>::VectorType stepMM,
                  typename rtk::RayBoxIntersectionFunction<CoordRepType, TOutputImage::ImageDimension>::VectorType sourcePosition,
                  typename rtk::RayBoxIntersectionFunction<CoordRepType, TOutputImage::ImageDimension>::VectorType dirVox,
                  typename rtk::RayBoxIntersectionFunction<CoordRepType, TOutputImage::ImageDimension>::VectorType np,
                  typename rtk::RayBoxIntersectionFunction<CoordRepType, TOutputImage::ImageDimension>::VectorType fp);

private:
  JosephForwardProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented

  // Functors
  TInterpolationWeightMultiplication m_InterpolationWeightMultiplication;
  TProjectedValueAccumulation        m_ProjectedValueAccumulation;
};

template <>
void
rtk::JosephForwardProjectionImageFilter<itk::VectorImage<double, 3>,
                                        itk::VectorImage<double, 3>,
                                        Functor::VectorInterpolationWeightMultiplication<double, double, itk::VariableLengthVector<double>>,
                                        Functor::VectorProjectedValueAccumulation<itk::VariableLengthVector<double>, itk::VariableLengthVector<double> > >
::Accumulate(ThreadIdType threadId,
                            rtk::ProjectionsRegionConstIteratorRayBased<itk::VectorImage<double, 3> >* itIn,
                            itk::ImageRegionIteratorWithIndex<itk::VectorImage<double, 3> > itOut,
                            itk::VariableLengthVector<double> sum,
                            rtk::RayBoxIntersectionFunction<double, 3>::VectorType stepMM,
                            rtk::RayBoxIntersectionFunction<double, 3>::VectorType sourcePosition,
                            rtk::RayBoxIntersectionFunction<double, 3>::VectorType dirVox,
                            rtk::RayBoxIntersectionFunction<double, 3>::VectorType np,
                            rtk::RayBoxIntersectionFunction<double, 3>::VectorType fp);

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkJosephForwardProjectionImageFilter.hxx"
#endif

#endif
