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
 * \ingroup ImageToImageFilter
 */

template <class TInputImage, class TOutputImage = TInputImage>
class ITK_EXPORT WaterPrecorrectionImageFilter:
    public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef WaterPrecorrectionImageFilter                     Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer< Self >                         Pointer;
  typedef itk::SmartPointer< const Self >                   ConstPointer;

  /** Convenient typedefs. */
  typedef typename TOutputImage::RegionType OutputImageRegionType;
  typedef std::vector< double >             VectorType;

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
  ~WaterPrecorrectionImageFilter() {}

  void ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId) ITK_OVERRIDE;

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
