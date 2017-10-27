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
 * \ingroup InPlaceImageFilter
 */
template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT BoellaardScatterCorrectionImageFilter :
  public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef BoellaardScatterCorrectionImageFilter              Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                                     InputImageType;
  typedef TOutputImage                                    OutputImageType;
  typedef typename OutputImageType::RegionType            OutputImageRegionType;

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
  ~BoellaardScatterCorrectionImageFilter() {}

  /** Requires full projection images to estimate scatter */
  void EnlargeOutputRequestedRegion(itk::DataObject *itkNotUsed(output)) ITK_OVERRIDE;
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

  /** Split the output's RequestedRegion into "num" pieces, returning
   * region "i" as "splitRegion". Reimplemented from ImageSource to ensure
   * that each thread covers entire projections. */
  unsigned int SplitRequestedRegion(unsigned int i, unsigned int num, OutputImageRegionType& splitRegion) ITK_OVERRIDE;
  virtual int SplitRequestedRegion(int i, int num, OutputImageRegionType& splitRegion);

private:
  BoellaardScatterCorrectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);               //purposely not implemented

  /** Air threshold on projection images. */
  double m_AirThreshold;

  /** Scatter to primary ratio */
  double m_ScatterToPrimaryRatio;

  /** Non-negativity constraint threshold */
  double m_NonNegativityConstraintThreshold;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkBoellaardScatterCorrectionImageFilter.hxx"
#endif

#endif
