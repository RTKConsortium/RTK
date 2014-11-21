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
#include <algorithm>

namespace rtk
{

template < unsigned char bitShift >
I0EstimationProjectionFilter<bitShift >::I0EstimationProjectionFilter()
{
	m_NBins = (unsigned int)(1 << (16 - bitShift));
	
	m_histogram.resize(m_NBins, 0);
	m_pastI0.resize(3, 0);

	std::cout << "Histogram size: " << m_NBins << std::endl;

	m_Median = true;
	m_UseRLS = false;
	m_UseTurbo = false;

	m_mutex = itk::MutexLock::New();

	m_I0 = 0;
	m_I0fwhm = 0;
	m_Np = 0;
	m_I0mean = 0.;
	m_Lambda = 1.0;
	m_dynThreshold = 20;

}

template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::BeforeThreadedGenerateData()
{
	std::vector<unsigned>::iterator it = m_histogram.begin();
	for (; it != m_histogram.end(); ++it)
		*it = 0;

	std::cout << "Number of threads : "<<this->GetNumberOfThreads() << std::endl;

	m_Nthreads = this->GetNumberOfThreads();

	ImageType::RegionType fullRegion = this->GetInput()->GetLargestPossibleRegion();
	
	m_Barrier = itk::Barrier::New();
	m_Barrier->Initialize(m_Nthreads);

	m_nsync = 0;
	}

template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId)
{
	std::vector<unsigned > m_thHisto;  // Per-thread histogram
	m_thHisto.resize(m_NBins, 0);

	itk::ImageRegionConstIterator<ImageType> itIn(this->GetInput(), outputRegionForThread);
	
	// Fill in its own histogram
	itIn.GoToBegin();
	while (!itIn.IsAtEnd() ) {
		m_thHisto[itIn.Get() >> bitShift]++;
		++itIn;
		if (m_UseTurbo) ++itIn;
	}
		
	m_mutex->Lock();
	{
		// Add its results to shared histogram
		for (unsigned int i = 0; i < m_NBins; ++i) {
			m_histogram[i] += m_thHisto[i];
		}
		
		// The last thread has to do something more
		++m_nsync;
		if (m_nsync >= m_Nthreads) {  

			// Search for upper bound of the histogram : gives the highest intensity value
			m_Imax = m_NBins - 1;
			while ((m_histogram[m_Imax] <= m_dynThreshold) && (m_Imax >= 0)) {
				--m_Imax;
			}
			while ((m_histogram[m_Imax] == 0) && (m_Imax < m_NBins)) {  // Get back
				++m_Imax;
			}

			// Search for lower bound of the histogram: gives the lowest intensity value
			m_Imin = 0;
			while ((m_histogram[m_Imin] <= m_dynThreshold) && (m_Imin <= m_Imax)) {
				++m_Imin;
			}
			while ((m_histogram[m_Imin] == 0) && (m_Imin >= 0)) {  // Get back
				--m_Imin;
			}
			m_Imin

			// Compute histogram dynamic 
			m_Irange = m_Imax - m_Imin;
			m_dynamicUsage = 100.*float(m_Irange) / float(m_NBins);




			// If Imax near zeros - problem to be fixed
			// If Imin near Imax - problem to be fixed

			// If Imax very close to m_NBins then possible saturation
			// If Imin far from 0 - high dark or very low attenuating objects




			int sum = 0;
			for (unsigned int i = m_Imin; i < m_Imax; ++i) {
				sum += m_histogram[i];
			}

		//	std::cout << "Number of pixels " << sum << std::endl;
		//	std::cout << "Last thread computing min/max : " << m_Imin << " " << m_Imax << " " << m_Irange << std::endl;

		}
	}
	m_mutex->Unlock();

	m_Barrier->Wait();
	// This threads knows that every other thread has put its contribution to the histogram
	
	/*
	m_mutex->Lock();
	{
		// Copy shared histogram back into its own only on valid range
		for (unsigned int i = m_Imin; i < m_Imax; ++i) {
			m_thHisto[i] = m_histogram[i];
		}
	}
	m_mutex->Unlock();
	
	int thrange = int( float(m_Irange) / float(m_Nthreads))+1 ;
	int imin = m_Imin + int(threadId)*thrange;
	int imax = imin+thrange;

	m_mutex->Lock();
	std::cout << (threadId) << " continues processing on " <<imin<<" "<<imax<< std::endl;
	m_mutex->Unlock();
		

	for (int i = imin; i < imax; ++i) {

	}

	*/





}

template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::AfterThreadedGenerateData()
{
	
	std::cout << "Detector dynamic usage " << m_dynamicUsage << std::endl;

	// Search for a mode btw (min+max)/2 and max
	unsigned i = (m_Imax >> 1);
	unsigned short maxId = i;
	unsigned maxVal = m_histogram[maxId];
	while (i < m_Imax) {
		if (m_histogram[i] >= maxVal) {
			maxVal = m_histogram[i];
			maxId = i;
		}
		++i;
	}
	//std::cout << maxId << " " << i << " " << maxVal << std::endl;
	unsigned peakPos = unsigned((maxId + 1) << bitShift);
	
	unsigned short lowBound = maxId;
	while ((m_histogram[lowBound] > (maxVal >> 1)) && (lowBound > 0)) {
		lowBound--;
	}

	unsigned short highBound = maxId;
	while ((m_histogram[highBound] > (maxVal >> 1)) && (highBound < m_Imax)) {
		highBound++;
	}

	unsigned peakFwhm = ((highBound - lowBound) << bitShift);



	// Percentage of pixels in background: can be used to prevent/prepare occlusion
	unsigned Ntotal = 0;
	for (i = m_Imin; i < m_Imax; ++i)
		Ntotal += m_histogram[i];

	unsigned Nback = 0;
	for (i = (lowBound - peakFwhm); i < m_Imax; ++i)
		Nback += m_histogram[i];

	float percent = 100.0*(float)Nback / (float)Ntotal;
	std::cout << "Percentage : " << percent << std::endl;

	m_lowBound = lowBound;
	m_highBound = highBound;
	m_middle = (highBound + lowBound) >> 1;

	m_I0 = peakPos;

	if (m_Median) {
		// Circular swap: I0[] ejected - only used for median filtering
		m_pastI0[0] = m_pastI0[1];
		m_pastI0[1] = m_pastI0[2];
		m_pastI0[2] = peakPos;

		// Use the median value on the three last estimates
		if (m_Np >= 3) {
			unsigned short b1 = m_pastI0[0];
			unsigned short b2 = m_pastI0[1];
			unsigned short b3 = m_pastI0[2];
			if (b1 > b2) std::swap(b1, b2);
			if (b2 > b3) std::swap(b2, b3);
			if (b1 > b2) std::swap(b1, b2);
			m_I0 = b2;
		}
	}

	if (m_UseRLS) {
		m_I0rls = (m_Np > 1) ? double(m_I0rls*(1.0 - m_Lambda) + peakPos*m_Lambda) : m_I0;
		m_I0 = m_I0rls;
	}

	m_I0mean = double(m_I0*m_Np + peakPos) / double(m_Np + 1);

	
	
	m_I0fwhm = peakFwhm;
	
	++m_Np;
	
}

} // end namespace rtk
#endif

