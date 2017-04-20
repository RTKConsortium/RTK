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

#ifndef rtkLagCorrectionImageFilter_h
#define rtkLagCorrectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include <itkVector.h>
#include <vector>

#include "rtkConfiguration.h"
#include "rtkMacro.h"

namespace rtk
{

/** \class LagCorrectionImageFilter
 * \brief Classical Linear Time Invariant Lag correction
 *
 * Recursive correction algorithm for detector decay characteristics.
 * Based on [Hsieh, Proceedings of SPIE, 2000]
 *
 * The IRF (Impulse Response Function) is given by:
 *    \f$h(k)=b_0 \delta(k) + \sum_{n=1}^N b_n e^{-a_n k}\f$
 * where \f$k\f$ is the discrete time, \f$N\f$ is the model order (number of exponentials),
 * \f$\delta(k)\f$ is the impulse function and the \f${a_n, b_n}_{n=1:N}\f$ parameters are respectively the exponential rates and
 * lag coefficients to be provided. The sum of all $b_n$ must be normalized such that h(0) equals 1.
 *
 * The parameters are typically estimated from either RSRF (Rising Step RF) or FSRF (Falling Step RF) response functions.
 *
 * \test rtklagcorrectiontest.cxx
 *
 * \author Sebastien Brousmiche
 *
 * \ingroup ImageToImageFilter
*/

template< typename TImage, unsigned ModelOrder >
class LagCorrectionImageFilter
: public itk::InPlaceImageFilter < TImage, TImage >
{
public:

  /** Standard class typedefs. */
  typedef LagCorrectionImageFilter                   Self;
  typedef itk::InPlaceImageFilter< TImage, TImage >  Superclass;
  typedef itk::SmartPointer< Self >                  Pointer;
  typedef itk::SmartPointer< const Self >            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(LagCorrectionImageFilter, ImageToImageFilter)

  typedef typename TImage::RegionType                  ImageRegionType;
  typedef typename TImage::SizeType                    ImageSizeType;
  typedef typename TImage::PixelType                   PixelType;
  typedef typename TImage::IndexType                   IndexType;
  typedef typename itk::Vector<float, ModelOrder>      VectorType;
  typedef typename std::vector<float>                  FloatVectorType;
  typedef typename TImage::RegionType                  OutputImageRegionType;

  /** Get / Set the model parameters A and B*/
  itkGetMacro(A, VectorType)
  itkGetMacro(B, VectorType)
  virtual void SetCoefficients(const VectorType A, const VectorType B)
  {
    if ((this->m_A != A) && (this->m_B != B))
    {
      if ((A.Size() == ModelOrder) && (B.Size() == ModelOrder)) {}
      this->m_A = A;
      this->m_B = B;
      this->Modified();
      m_NewParamJustReceived = true;
    }
  }

protected:
  LagCorrectionImageFilter();
  ~LagCorrectionImageFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void ThreadedGenerateData(const ImageRegionType & outputRegionForThread, itk::ThreadIdType threadId) ITK_OVERRIDE;

  /** The correction is applied along the third (stack) dimension.
      Therefore, we must avoid splitting along the stack.
      The split is done along the second dimension. */
  unsigned int SplitRequestedRegion(unsigned int i, unsigned int num, OutputImageRegionType& splitRegion) ITK_OVERRIDE;
  virtual int SplitRequestedRegion(int i, int num, OutputImageRegionType& splitRegion);

  VectorType m_A;           // a_n coefficients (lag rates)
  VectorType m_B;           // b coefficients (lag coefficients)
  VectorType m_ExpmA;       // exp(-a)
  float      m_SumB;        // normalization factor

protected:
  FloatVectorType m_S;                      // State variable

private:
  LagCorrectionImageFilter(const Self &); // purposely not implemented
  void operator=(const Self &);           // purposely not implemented

  bool            m_NewParamJustReceived;   // For state/correction initialization
  IndexType       m_StartIdx;               // To account for cropping
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkLagCorrectionImageFilter.hxx"
#endif

#endif
