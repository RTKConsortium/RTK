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
#include <itkImageRegionIterator.h>
#include <algorithm>

namespace rtk
{

template < unsigned char bitShift >
I0EstimationProjectionFilter<bitShift >::I0EstimationProjectionFilter()
{
	m_NBins = (unsigned int)(1 << (16 - bitShift));
	
	m_histogram.resize(m_NBins, 0);
	m_pastI0.resize(3, 0);

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

/*
template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::GenerateOutputInformation()
{
	std::cout << "Generate Output" << std::endl;
	// call the superclass' implementation of this method
	Superclass::GenerateOutputInformation();
	
	// Get pointers to the input and output
	InputImageConstPointer inputPtr = this->GetInput();
	OutputImagePointer     outputPtr = this->GetOutput();

	if (!outputPtr || !inputPtr) {
		return;
	}

	// Compute the output spacing, size, and start index
	const typename InputImageType::SpacingType &
		inputSpacing = inputPtr->GetSpacing();
	const typename InputImageType::SizeType &   inputSize =
		inputPtr->GetLargestPossibleRegion().GetSize();
	const typename InputImageType::IndexType &  inputStartIndex =
		inputPtr->GetLargestPossibleRegion().GetIndex();
	
	typename OutputImageType::SpacingType outputSpacing;
	outputSpacing[0] = float(1 << bitShift);
	outputSpacing[1] = 1.0;
	outputSpacing[2] = 1.0;

	outputPtr->SetSpacing(outputSpacing);
	
	typename OutputImageType::SizeType outputSize;
	unsigned idx = unsigned(float(m_NBins) / float(inputSize[0]));
	outputSize[0] = inputSize[0];
	outputSize[1] = idx;
	outputSize[2] = 1;

	typename OutputImageType::IndexType outputStartIndex;
	outputStartIndex[0] = 0;
	outputStartIndex[1] = 0;
	outputStartIndex[2] = inputStartIndex[2];

	// Set region
	typename OutputImageType::RegionType outputRegion;
	outputRegion.SetSize(outputSize);
	outputRegion.SetIndex(outputStartIndex);
	outputPtr->SetLargestPossibleRegion(outputRegion);

}

template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::PropagateRequestedRegion()
{
	std::cout << "Propagate region" << std::endl;
}

template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::GenerateInputRequestedRegion()
{	
	std::cout << "Generate Input req region" << std::endl;

	// Call the superclass' implementation of this method
	Superclass::GenerateInputRequestedRegion();

	// Get pointers to the input and output
	InputImagePointer  inputPtr = const_cast< InputImageType * >(this->GetInput());
	OutputImagePointer outputPtr = this->GetOutput();

	if (!inputPtr || !outputPtr) {
		return;
	}

	std::cout << inputPtr->GetRequestedRegion().GetIndex() << std::endl;
	std::cout << inputPtr->GetRequestedRegion().GetSize() << std::endl;
	
	const typename InputImageType::SpacingType &
		inputSpacing = inputPtr->GetSpacing();
	InputImageType::SizeType inputSize =
		inputPtr->GetLargestPossibleRegion().GetSize();
	const typename InputImageType::IndexType inputIndex =
		inputPtr->GetLargestPossibleRegion().GetIndex();
	
	inputSize[0] = 1421;
 inputSize[1] = 1420;

	typename InputImageType::RegionType inputRequestedRegion;
	inputRequestedRegion.SetIndex(inputIndex);
	inputRequestedRegion.SetSize(inputSize);

	inputPtr->SetLargestPossibleRegion(inputRequestedRegion);
	inputPtr->SetRequestedRegion(inputRequestedRegion);
	inputPtr->SetBufferedRegion(inputRequestedRegion);

	std::cout << inputPtr->GetRequestedRegion().GetIndex() << std::endl;
	std::cout << inputPtr->GetRequestedRegion().GetSize() << std::endl;
	std::cout << inputPtr->GetLargestPossibleRegion().GetSize() << std::endl;
	std::cout << inputPtr->GetBufferedRegion().GetSize() << std::endl;
	
}*/

template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::BeforeThreadedGenerateData()
{
	std::vector<unsigned>::iterator it = m_histogram.begin();
	for (; it != m_histogram.end(); ++it)
		*it = 0;

	m_Nthreads = this->GetNumberOfThreads();

	InputImageType::RegionType fullRegion = this->GetInput()->GetLargestPossibleRegion();
	
	m_Barrier = itk::Barrier::New();
	m_Barrier->Initialize(m_Nthreads);

	m_nsync = 0;
	}

template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId)
{
	std::vector<unsigned > m_thHisto;  // Per-thread histogram
	m_thHisto.resize(m_NBins, 0);

	itk::ImageRegionConstIterator<InputImageType> itIn(this->GetInput(), outputRegionForThread);
	
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
			//m_Imin

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

	//m_mutex->Lock();
	//std::cout << (threadId) << " continues processing on " <<imin<<" "<<imax<< std::endl;
	//m_mutex->Unlock();
		

	for (int i = imin; i < imax; ++i) {

	}

	


	
}

template < unsigned char bitShift >
void I0EstimationProjectionFilter<bitShift >::AfterThreadedGenerateData()
{
	// Add its results to shared histogram
	itk::ImageRegionIterator<OutputImageType> itOut(this->GetOutput(), this->GetOutput()->GetLargestPossibleRegion());
	unsigned idx = 0;
	itOut.GoToBegin();
	while (!itOut.IsAtEnd()) {
		itOut.Set(0);
		++itOut;
	}
	itOut.GoToBegin();
	idx = 0;
	while ((idx<(m_NBins - 1)) && (!itOut.IsAtEnd())) {
		itOut.Set((unsigned int)(m_histogram[++idx]));
		++itOut;
	}


	std::cout << "Detector dynamic usage " << m_dynamicUsage << std::endl;






	// Search for the background mode in the last quarter of the histogram
	unsigned startIdx = m_Imin + 3 * ((m_Imax + m_Imin) >> 2);
	unsigned short i = startIdx;
	unsigned short maxId = startIdx;
	unsigned maxVal = m_histogram[startIdx];
	while (i < m_Imax) {
		if (m_histogram[i] >= maxVal) {
			maxVal = m_histogram[i];
			maxId = i;
		}
		++i;
	}
	m_I0 = unsigned((maxId + 1) << bitShift);
	// If estimated I0 at the boundaries, either startIdx or Imax then we missed smth or no background mode
	
	unsigned short widthval = unsigned short(float(maxVal>>1));
	unsigned short lowBound = maxId;
	while ((m_histogram[lowBound] > widthval) && (lowBound > 0)) {
		lowBound--;
	}

	unsigned short highBound = maxId;
	while ((m_histogram[highBound] > widthval) && (highBound < m_Imax)) {
		highBound++;
	}

	unsigned peakFwhm = ((highBound - lowBound) << bitShift);
	m_I0fwhm = peakFwhm;
	m_I0sigma = float(m_I0fwhm) / 2.3548f;

	// Percentage of pixels in background: can be used to prevent/prepare occlusion
	unsigned Ntotal = 0;
	for (i = m_Imin; i < m_Imax; ++i)
		Ntotal += m_histogram[i];

	// Computes number of background pixels
	unsigned Nback = 0;
	for (i = (lowBound - peakFwhm); i < m_Imax; ++i)
		Nback += m_histogram[i];

	float percent = 100.0*(float)Nback / (float)Ntotal;
	std::cout << "Percentage : " << percent << std::endl;

	m_lowBound = lowBound;
	m_highBound = highBound;
	m_middle = (highBound + lowBound) >> 1;

	// Update bounds
	float lbd = 0.05;
	m_lowBndRls = (m_Np > 1) ? float(m_lowBound*(1.0 - lbd) + float(m_lowBndRls)*lbd) : m_lowBound;
	m_highBndRls = (m_Np > 1) ? float(m_highBound*(1.0 - lbd) + float(m_highBndRls)*lbd) : m_highBound;
	m_middleRls = (m_Np > 1) ? float(m_middle*(1.0 - lbd) + float(m_middleRls)*lbd) : m_middle;


	

	if (m_Median) {
		// Circular swap: I0[] ejected - only used for median filtering
		m_pastI0[0] = m_pastI0[1];
		m_pastI0[1] = m_pastI0[2];
		m_pastI0[2] = m_I0;

		// Use the median value on the three last estimates
		if (m_Np >= 3) {
			unsigned short b1 = m_pastI0[0];
			unsigned short b2 = m_pastI0[1];
			unsigned short b3 = m_pastI0[2];
			if (b1 > b2) std::swap(b1, b2);
			if (b2 > b3) std::swap(b2, b3);
			if (b1 > b2) std::swap(b1, b2);
			m_I0median = b2;
		}
	} else {
		m_I0median = m_I0;
	}

	if (m_UseRLS) {
		m_I0rls = (m_Np > 1) ? float(m_I0rls*(1.0 - m_Lambda) + float(m_I0median)*m_Lambda) : float(m_I0median);
	} else {
		m_I0rls = m_I0;
	}

	m_I0mean = unsigned short(float(m_I0median*m_Np + m_I0) / float(m_Np + 1));
		
	
	
	++m_Np;

	std::cout << "I0 " << m_I0 << std::endl;
	std::cout << "N " << m_NBins << std::endl;
	std::cout << "Np                 " << m_Np << std::endl;

}

} // end namespace rtk
#endif

