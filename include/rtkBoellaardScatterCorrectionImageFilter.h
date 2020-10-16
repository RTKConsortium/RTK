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

#ifndef rtkBoellaardScatterCorrectionImageFilter_h
#define rtkBoellaardScatterCorrectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkConfiguration.h"

namespace rtk
{

/** \class BoellaardScatterCorrectionImageFilter
 * \brief Scatter correction for cone-beam CT reconstruction.
 *
 * The scatter correction algorithm is based on the work of
 * [Boellaard, Rad Onc, 1997]. It assumes a homogeneous contribution of scatter
 * which is computed depending on the amount of tissues traversed by x-rays.
 *
 * \author Simon Rit
 *
 * \ingroup RTK InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage = TInputImage>
class ITK_EXPORT BoellaardScatterCorrectionImageFilter : public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(BoellaardScatterCorrectionImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(BoellaardScatterCorrectionImageFilter);
#endif

  /** Standard class type alias. */
  using Self = BoellaardScatterCorrectionImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using OutputImageRegionType = typename OutputImageType::RegionType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(BoellaardScatterCorrectionImageFilter, itk::ImageToImageFilter);

  /** Get / Set the air threshold on projection images. The threshold is used to
   ** evaluate which part of the x-rays have traversed the patient. */
  itkGetMacro(AirThreshold, double);
  itkSetMacro(AirThreshold, double);

  /** Get / Set the scatter-to-primary ratio on projection images. This is used
   ** to measure the scatter level on the projection image from the total measure
   ** signal. */
  itkGetMacro(ScatterToPrimaryRatio, double);
  itkSetMacro(ScatterToPrimaryRatio, double);

  /** Get / Set the non-negativity constraint threshold. The pixels values will
  ** no be alowed below this signal to avoid nan when computing the log. */
  itkGetMacro(NonNegativityConstraintThreshold, double);
  itkSetMacro(NonNegativityConstraintThreshold, double);

protected:
  BoellaardScatterCorrectionImageFilter();
  ~BoellaardScatterCorrectionImageFilter() override = default;

  /** Requires full projection images to estimate scatter */
  void
  EnlargeOutputRequestedRegion(itk::DataObject * itkNotUsed(output)) override;
  void
  ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId) override;

  /** Split the output's RequestedRegion into "num" pieces, returning
   * region "i" as "splitRegion". Reimplemented from ImageSource to ensure
   * that each thread covers entire projections. */
  unsigned int
  SplitRequestedRegion(unsigned int i, unsigned int num, OutputImageRegionType & splitRegion) override;
  virtual int
  SplitRequestedRegion(int i, int num, OutputImageRegionType & splitRegion);

private:
  /** Air threshold on projection images. */
  double m_AirThreshold{ 32000 };

  /** Scatter to primary ratio */
  double m_ScatterToPrimaryRatio{ 0. };

  /** Non-negativity constraint threshold */
  double m_NonNegativityConstraintThreshold{ 20 };
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkBoellaardScatterCorrectionImageFilter.hxx"
#endif

#endif
