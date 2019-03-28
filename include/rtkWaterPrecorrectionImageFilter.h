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

#ifndef rtkWaterPrecorrectionImageFilter_h
#define rtkWaterPrecorrectionImageFilter_h

#include <vector>
#include <itkInPlaceImageFilter.h>

#include "rtkConfiguration.h"

namespace rtk
{
/** \class WaterPrecorrectionImageFilter
 * \brief Performs the classical water precorrection for beam hardening (Kachelriess, Med. Phys. 2006)
 *
 * \test rtkwaterprecorrectiontest.cxx
 *
 * \author S. Brousmiche
 *
 * \ingroup RTK ImageToImageFilter
 */

template <class TInputImage, class TOutputImage = TInputImage>
class ITK_EXPORT WaterPrecorrectionImageFilter:
    public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class type alias. */
  using Self = WaterPrecorrectionImageFilter;
  using Superclass = itk::InPlaceImageFilter<TInputImage,TOutputImage>;
  using Pointer = itk::SmartPointer< Self >;
  using ConstPointer = itk::SmartPointer< const Self >;

  /** Convenient type alias. */
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using VectorType = std::vector< double >;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(WaterPrecorrectionImageFilter, ImageToImageFilter);

  /** Get / Set the Median window that are going to be used during the operation
    */
  itkGetMacro(Coefficients, VectorType);
  virtual void SetCoefficients (const VectorType _arg)
    {
    if ( this->m_Coefficients != _arg )
      {
      this->m_Coefficients = _arg;
      this->Modified();
      }
    }

protected:
  WaterPrecorrectionImageFilter();
  ~WaterPrecorrectionImageFilter() override = default;

#if ITK_VERSION_MAJOR<5
  void ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId) override;
#else
  void DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;
#endif

private:
  WaterPrecorrectionImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);                //purposely not implemented

  VectorType m_Coefficients;      // Correction coefficients
};
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkWaterPrecorrectionImageFilter.hxx"
#endif

#endif
