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

#ifndef __rtkI0EstimationProjectionFilter_txx
#define __rtkI0EstimationProjectionFilter_txx

#include <itkImageRegionConstIterator.h>

namespace rtk
{

template < unsigned char bitShift >
I0EstimationProjectionFilter<bitShift >::I0EstimationProjectionFilter()
{
	m_NBins = (unsigned int)(1 << (16 - bitShift));
	std::cout << "Histogram size: "<< m_NBins << std::endl;

	m_mutex = itk::MutexLock::New();
}

/*template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::GenerateOutputInformation()
{
	// call the superclass' implementation of this method
	Superclass::GenerateOutputInformation();

	HistogramType::Pointer outputPtr = const_cast<HistogramType *>(this->GetOutput());
	
	HistogramType::RegionType outputRegion = outputPtr->GetLargestPossibleRegion();
	HistogramType::SizeType size;
	size[0] = m_NBins;
	size[1] = 1;
	size[2] = 1;
	outputRegion.SetSize(size);
	HistogramType::IndexType start;
	start[0] = 0;
	start[1] = 0;
	start[2] = 0;
	outputRegion.SetIndex(start);
	this->GetOutput()->SetLargestPossibleRegion(outputRegion);
	this->GetOutput()->Allocate();

	std::cout << this->GetOutput()->GetLargestPossibleRegion() << std::endl;
}*/

template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::BeforeThreadedGenerateData()
{
	m_histogram.clear();
	m_histogram.resize(m_NBins, 0);
}

template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType itkNotUsed(threadId))
{
	std::vector<unsigned > m_thHisto;  // Per-thread histogram
	m_thHisto.resize(m_NBins, 0);

	itk::ImageRegionConstIterator<ImageType> itIn(this->GetInput(), outputRegionForThread);

	// Fill histogram
	itIn.GoToBegin();
	while (!itIn.IsAtEnd() ) {
		m_thHisto[itIn.Get() >> bitShift]++;
		++itIn;
	}
	
	// Add 
	m_mutex->Lock();
	for (unsigned int i = 0; i < m_NBins; ++i) {
		m_histogram[i] += m_thHisto[i];
	}
	m_mutex->Unlock();
}

template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::AfterThreadedGenerateData()
{
	unsigned int maxhh = 0;
	unsigned int idxmax = 0;
	for (unsigned int i = 0; i < m_NBins; ++i) {
		if (m_histogram[i] >= maxhh) {
			maxhh = m_histogram[i];
			idxmax = i;
		}
	}
	std::cout << maxhh << " " << idxmax << " " << (idxmax << bitShift) << std::endl;
	
	// Search for upper bound
	unsigned maxh = m_NBins - 1;
	while ((m_histogram[maxh] == 0) && (maxh >= 0)) {
		maxh--;
	}
	std::cout << "Upper bound: " << maxh << std::endl;

	// Search for a mode btw (min+max)/2 and max
	//unsigned maxVal = 0;
	unsigned i = (maxh >> 1);
	unsigned short maxId = i;
	unsigned maxVal = m_histogram[maxId];
	while (i<maxh) {
		if (m_histogram[i]>=maxVal) {
			maxVal = m_histogram[i];
			maxId = i;
		}
		++i;
	}
	//std::cout << maxId << " " << i << " " << maxVal << std::endl;
	unsigned peakPos = (unsigned)((maxId+1) << bitShift);

	unsigned short lowBound = maxId;
	while ((m_histogram[lowBound]>(maxVal >> 1)) && (lowBound>0)) {
		lowBound--;
	}

	unsigned short highBound = maxId;
		while ((m_histogram[highBound]>(maxVal >> 1)) && (highBound<maxh)) {
		highBound++;
	}

	unsigned peakFwhm = ((highBound - lowBound) << bitShift);
	std::cout << "Peak at " << peakPos << " with FWHM "<<peakFwhm<< std::endl;

}

} // end namespace rtk
#endif
