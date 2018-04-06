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

  inline TOutput operator()( const ThreadIdType itkNotUsed(threadId),
                             const double itkNotUsed(stepLengthInVoxel),
                             const TCoordRepType weight,
                             const TInput *p,
                             const int i ) const
  {
    return weight*p[i];
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
          class TInterpolationWeightMultiplication = Functor::InterpolationWeightMultiplication<typename TInputImage::PixelType, double>,
          class TProjectedValueAccumulation        = Functor::ProjectedValueAccumulation<typename TInputImage::PixelType, typename TOutputImage::PixelType>
          >
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
  typedef typename TOutputImage::PixelType                       OutputPixelType;
  typedef typename TOutputImage::RegionType                      OutputImageRegionType;
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

  inline OutputPixelType BilinearInterpolation(const ThreadIdType threadId,
                                               const double stepLengthInVoxel,
                                               const InputPixelType *pxiyi,
                                               const InputPixelType *pxsyi,
                                               const InputPixelType *pxiys,
                                               const InputPixelType *pxsys,
                                               const double x,
                                               const double y,
                                               const int ox,
                                               const int oy);

  inline OutputPixelType BilinearInterpolationOnBorders(const ThreadIdType threadId,
                                               const double stepLengthInVoxel,
                                               const InputPixelType *pxiyi,
                                               const InputPixelType *pxsyi,
                                               const InputPixelType *pxiys,
                                               const InputPixelType *pxsys,
                                               const double x,
                                               const double y,
                                               const int ox,
                                               const int oy,
                                               const double minx,
                                               const double miny,
                                               const double maxx,
                                               const double maxy);

private:
  JosephForwardProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented

  TInterpolationWeightMultiplication m_InterpolationWeightMultiplication;
  TProjectedValueAccumulation        m_ProjectedValueAccumulation;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkJosephForwardProjectionImageFilter.hxx"
#endif

#endif
