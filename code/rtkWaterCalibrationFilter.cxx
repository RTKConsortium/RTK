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

#ifndef __rtkWaterCalibrationFilter_cxx
#define __rtkWaterCalibrationFilter_cxx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkWaterPrecorrectionFilter.h"

typedef itk::Image<unsigned short, 2> TImage;

namespace rtk
{

	WaterCalibrationFilter::WaterCalibrationFilter()
	{
	}

	void WaterCalibrationFilter
		::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId)
	{
		itk::ImageRegionConstIterator<TImage> itIn(this->GetInput(), outputRegionForThread);
		itk::ImageRegionIterator<TImage>      itOut(this->GetOutput(), outputRegionForThread);

		itIn.GoToBegin();
		itOut.GoToBegin();
		while (!itIn.IsAtEnd()){
			float v = itIn.Get();
			itOut.Set( std::powf(v, m_Order) );
			++itIn;
			++itOut;
		}


	}
} // end namespace rtk

#endif // __rtkWaterCalibrationFilter_cxx
