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

#ifndef __rtkWaterPrecorrectionFilter_cxx
#define __rtkWaterPrecorrectionFilter_cxx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkWaterPrecorrectionFilter.h"

typedef itk::Image<unsigned short, 2> TImage;

namespace rtk
{

	template <unsigned int modelOrder>
	WaterPrecorrectionFilter<modelOrder>::WaterPrecorrectionFilter()
	{
	}

	template <unsigned int modelOrder>
	void WaterPrecorrectionFilter<modelOrder>
		::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId)
	{
		int csize = modelOrder;

		itk::ImageRegionConstIterator<TImage> itIn(this->GetInput(), outputRegionForThread);
		itk::ImageRegionIterator<TImage>     itOut(this->GetOutput(), outputRegionForThread);

		if (csize >= 3)
		{
			itIn.GoToBegin();
			itOut.GoToBegin();

			while (!itIn.IsAtEnd()) {
				float v = itIn.Get();
				float out = m_Coefficients[0] + m_Coefficients[1] * v;
				float bpow = v * v;

				for (int i = 2; i < csize; i++) {
					out += m_Coefficients[i] * bpow;
					bpow = bpow*v;
				}
				itOut.Set(out);

				++itIn;
				++itOut;
			}
		}
		else if ((csize == 2) && ((m_Coefficients[0] != 0) || (m_Coefficients[1] != 1)))
		{
			itIn.GoToBegin();
			itOut.GoToBegin();
			while (!itIn.IsAtEnd()) {
				itOut.Set(m_Coefficients[0] + m_Coefficients[1] * itIn.Get());
				++itIn;
				++itOut;
			}
		}
		else if ((csize == 1) && (m_Coefficients[0] != 0))
		{
			itIn.GoToBegin();
			itOut.GoToBegin();
			while (!itIn.IsAtEnd()) {
				itOut.Set(m_Coefficients[0] + itIn.Get());
			}
		}
	}
} // end namespace rtk

#endif // __rtkWaterPrecorrectionFilter_cxx
