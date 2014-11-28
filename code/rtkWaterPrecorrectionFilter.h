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

#ifndef __rtkWaterPrecorrectionFilter_h
#define __rtkWaterPrecorrectionFilter_h

#include <vector>
#include <itkImageToImageFilter.h>

#include "rtkConfiguration.h"

namespace rtk
{

/** \class WaterPrecorrectionFilter
 * \brief Performs the classical water precorrection for beam hardening (Kachelriess, Med. Phys. 2006)
 *
 * \test rtkwaterprecorrectiontest.cxx
 *
 * \author S. Brousmiche
 *
 * \ingroup ImageToImageFilter
 */

template <unsigned int modelOrder>
class ITK_EXPORT WaterPrecorrectionFilter :
	public itk::ImageToImageFilter< itk::Image<float, 2>, itk::Image<float, 2> >
{
public:
	typedef itk::Image<float, 2>              TImage;

  /** Standard class typedefs. */
	typedef WaterPrecorrectionFilter                Self;
  typedef itk::ImageToImageFilter<TImage,TImage>  Superclass;
  typedef itk::SmartPointer<Self>                 Pointer;
  typedef itk::SmartPointer<const Self>           ConstPointer;
	
	typedef itk::Vector<float, modelOrder> VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
	itkTypeMacro(WaterPrecorrectionFilter, ImageToImageFilter);

  /** Get / Set the Median window that are going to be used during the operation */
	itkGetMacro(Coefficients, VectorType);
	itkSetMacro(Coefficients, VectorType);

protected:
	WaterPrecorrectionFilter();
	virtual ~WaterPrecorrectionFilter() {};

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

private:
	WaterPrecorrectionFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

	VectorType m_Coefficients;      // Correction coefficients
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkWaterPrecorrectionFilter.cxx"
#endif

#endif
