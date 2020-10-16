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
 * \f$\delta(k)\f$ is the impulse function and the \f${a_n, b_n}_{n=1:N}\f$ parameters are respectively the exponential
 * rates and lag coefficients to be provided. The sum of all $b_n$ must be normalized such that h(0) equals 1.
 *
 * The parameters are typically estimated from either RSRF (Rising Step RF) or FSRF (Falling Step RF) response
 * functions.
 *
 * \test rtklagcorrectiontest.cxx
 *
 * \author Sebastien Brousmiche
 *
 * \ingroup RTK ImageToImageFilter
 */

template <typename TImage, unsigned VModelOrder>
class LagCorrectionImageFilter : public itk::InPlaceImageFilter<TImage, TImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(LagCorrectionImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(LagCorrectionImageFilter);
#endif

  /** Standard class type alias. */
  using Self = LagCorrectionImageFilter;
  using Superclass = itk::InPlaceImageFilter<TImage, TImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(LagCorrectionImageFilter, ImageToImageFilter);

  using ImageRegionType = typename TImage::RegionType;
  using ImageSizeType = typename TImage::SizeType;
  using PixelType = typename TImage::PixelType;
  using IndexType = typename TImage::IndexType;
  using VectorType = typename itk::Vector<float, VModelOrder>;
  using FloatVectorType = typename std::vector<float>;
  using OutputImageRegionType = typename TImage::RegionType;

  /** Get / Set the model parameters A and B*/
  itkGetMacro(A, VectorType);
  itkGetMacro(B, VectorType);
  virtual void
  SetCoefficients(const VectorType A, const VectorType B)
  {
    if ((this->m_A != A) && (this->m_B != B))
    {
      if ((A.Size() == VModelOrder) && (B.Size() == VModelOrder))
      {
      }
      this->m_A = A;
      this->m_B = B;
      this->Modified();
      m_NewParamJustReceived = true;
    }
  }

protected:
  LagCorrectionImageFilter();
  ~LagCorrectionImageFilter() override = default;

  void
  GenerateOutputInformation() override;

  void
  GenerateInputRequestedRegion() override;

  void
  ThreadedGenerateData(const ImageRegionType & thRegion, itk::ThreadIdType threadId) override;

  /** The correction is applied along the third (stack) dimension.
      Therefore, we must avoid splitting along the stack.
      The split is done along the second dimension. */
  unsigned int
  SplitRequestedRegion(unsigned int i, unsigned int num, OutputImageRegionType & splitRegion) override;
  virtual int
  SplitRequestedRegion(int i, int num, OutputImageRegionType & splitRegion);

  VectorType m_A;     // a_n coefficients (lag rates)
  VectorType m_B;     // b coefficients (lag coefficients)
  VectorType m_ExpmA; // exp(-a)
  float      m_SumB;  // normalization factor

protected:
  FloatVectorType m_S; // State variable

private:
  bool      m_NewParamJustReceived; // For state/correction initialization
  IndexType m_StartIdx;             // To account for cropping
};

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkLagCorrectionImageFilter.hxx"
#endif

#endif
