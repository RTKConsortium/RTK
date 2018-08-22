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

#ifndef rtkJosephForwardAttenuatedProjectionImageFilter_h
#define rtkJosephForwardAttenuatedProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkMacro.h"
#include <itkPixelTraits.h>
#include <math.h>
#include <vector>

namespace rtk
{
namespace Functor
{
/** \class InterpolationWeightMultiplicationAttenuated
 * \brief Function to multiply the interpolation weights with the projected
 * volume values and attenuation map.
 *
 * \author Antoine Robert
 *
 * \ingroup Functions
 */
template< class TInput, class TCoordRepType, class TOutput = TInput >
class InterpolationWeightMultiplicationAttenuated
{
public:
  InterpolationWeightMultiplicationAttenuated()
  {
    for (int i = 0; i < ITK_MAX_THREADS; i++)
      {
      m_AttenuationRay[i] = 0;
      m_AttenuationPixel[i] = 0;
      m_ex1[i] = 1;
      }
  }

  ~InterpolationWeightMultiplicationAttenuated() {}
  bool operator!=( const InterpolationWeightMultiplicationAttenuated & ) const {
    return false;
  }

  bool operator==(const InterpolationWeightMultiplicationAttenuated & other) const
  {
    return !( *this != other );
  }

  inline TOutput operator()( const ThreadIdType threadId,
                             const double stepLengthInVoxel,
                             const TCoordRepType weight,
                             const TInput *p,
                             const int i )
  {
    const double w = weight*stepLengthInVoxel;

    m_AttenuationRay[threadId] += w*(p+m_AttenuationMinusEmissionMapsPtrDiff)[i];
    m_AttenuationPixel[threadId] += w*(p+m_AttenuationMinusEmissionMapsPtrDiff)[i];
    return weight*p[i];
  }

  void SetAttenuationMinusEmissionMapsPtrDiff(std::ptrdiff_t pd) {m_AttenuationMinusEmissionMapsPtrDiff = pd;}
  TOutput * GetAttenuationRay() {return m_AttenuationRay;}
  TOutput * GetAttenuationPixel() {return m_AttenuationPixel;}
  TOutput * GetEx1() {return m_ex1;}

private:
  std::ptrdiff_t m_AttenuationMinusEmissionMapsPtrDiff;
  TInput m_AttenuationRay[ITK_MAX_THREADS];
  TInput m_AttenuationPixel[ITK_MAX_THREADS];
  TInput m_ex1[ITK_MAX_THREADS];
};

/** \class ComputeAttenuationCorrection
 * \brief Function to compute the attenuation correction on the projection.
 *
 * \author Antoine Robert
 *
 * \ingroup Functions
 */
template< class TInput, class TOutput>
class ComputeAttenuationCorrection
{
public:
  typedef itk::Vector<double, 3> VectorType;

  ComputeAttenuationCorrection(){}
  ~ComputeAttenuationCorrection() {}
  bool operator!=( const ComputeAttenuationCorrection & ) const
  {
    return false;
  }

  bool operator==(const ComputeAttenuationCorrection & other) const
  {
    return !( *this != other );
  }

  inline TOutput operator()(const ThreadIdType threadId,
                            const TInput volumeValue,
                            const VectorType &stepInMM)
  {
    TInput ex2 = exp(-m_AttenuationRay[threadId]*stepInMM.GetNorm() );
    TInput wf;

    if(m_AttenuationPixel[threadId] > 0)
      {
      wf = (m_ex1[threadId]-ex2)/m_AttenuationPixel[threadId];
      }
    else
      {
      wf  = m_ex1[threadId]*stepInMM.GetNorm();
      }

    m_ex1[threadId] = ex2 ;
    m_AttenuationPixel[threadId] = 0;
    return wf *volumeValue;
  }

  void SetAttenuationRayVector( TInput *attenuationRayVector) {m_AttenuationRay = attenuationRayVector;}
  void SetAttenuationPixelVector( TInput *attenuationPixelVector) {m_AttenuationPixel = attenuationPixelVector;}
  void SetEx1( TInput *ex1) {m_ex1 = ex1;}

private:
  TInput* m_AttenuationRay;
  TInput* m_AttenuationPixel;
  TInput* m_ex1;
};

/** \class ProjectedValueAccumulationAttenuated
 * \brief Function to accumulate the ray casting on the projection.
 *
 * \author Antoine Robert
 *
 * \ingroup Functions
 */
template< class TInput, class TOutput >
class ProjectedValueAccumulationAttenuated
{
public:
  typedef itk::Vector<double, 3> VectorType;

  ProjectedValueAccumulationAttenuated() {}
  ~ProjectedValueAccumulationAttenuated() {}
  bool operator!=( const ProjectedValueAccumulationAttenuated & ) const
  {
    return false;
  }

  bool operator==(const ProjectedValueAccumulationAttenuated & other) const
  {
    return !( *this != other );
  }

  inline void operator()( const ThreadIdType threadId,
                          const TInput &input,
                          TOutput &output,
                          const TOutput &rayCastValue,
                          const VectorType &stepInMM,
                          const VectorType &itkNotUsed(source),
                          const VectorType &itkNotUsed(sourceToPixel),
                          const VectorType &itkNotUsed(nearestPoint),
                          const VectorType &itkNotUsed(farthestPoint) )
  {
    output = input + rayCastValue ;
    m_Attenuation[threadId] = 0;
    m_ex1[threadId] = 1;
  }

  void SetAttenuationVector(  TInput *attenuationVector) {m_Attenuation = attenuationVector;}
  void SetEx1( TInput *ex1) {m_ex1 = ex1;}

private:
  TInput* m_Attenuation;
  TInput* m_ex1;
};
} // end namespace Functor

/** \class JosephForwardAttenuatedProjectionImageFilter
 * \brief Joseph forward projection.
 *
 * Performs a attenuated Joseph forward projection, i.e. accumulation along x-ray lines,
 * using [Joseph, IEEE TMI, 1982] and [Gullberg, Phys. Med. Biol., 1985]. The forward projector tests if the  detector
 * has been placed after the source and the volume. If the detector is in the volume
 * the ray tracing is performed only until that point.
 *
 * \test rtkforwardattenuatedprojectiontest.cxx
 *
 * \author Antoine Robert
 *
 * \ingroup Projector
 */

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication = Functor::InterpolationWeightMultiplicationAttenuated<typename TInputImage::PixelType,typename itk::PixelTraits<typename TInputImage::PixelType>::ValueType>,
          class TProjectedValueAccumulation        = Functor::ProjectedValueAccumulationAttenuated<typename TInputImage::PixelType, typename TOutputImage::PixelType>,
          class TSumAlongRay     = Functor::ComputeAttenuationCorrection<typename TInputImage::PixelType, typename TOutputImage::PixelType>
          >
class ITK_EXPORT JosephForwardAttenuatedProjectionImageFilter :
    public JosephForwardProjectionImageFilter<TInputImage,TOutputImage,TInterpolationWeightMultiplication, TProjectedValueAccumulation, TSumAlongRay>
{
public:
  /** Standard class typedefs. */
  typedef JosephForwardAttenuatedProjectionImageFilter           Self;
  typedef JosephForwardProjectionImageFilter<TInputImage,TOutputImage,TInterpolationWeightMultiplication, TProjectedValueAccumulation, TSumAlongRay> Superclass;
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
  itkTypeMacro(JosephForwardAttenuatedProjectionImageFilter, JosephForwardProjectionImageFilter);

protected:
  JosephForwardAttenuatedProjectionImageFilter();
  virtual ~JosephForwardAttenuatedProjectionImageFilter() ITK_OVERRIDE {}

  /** Apply changes to the input image requested region. */
  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  /** Only the last two inputs should be in the same space so we need
   * to overwrite the method. */
  void VerifyInputInformation() ITK_OVERRIDE ;

private:
  JosephForwardAttenuatedProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented
};
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkJosephForwardAttenuatedProjectionImageFilter.hxx"
#endif

#endif
