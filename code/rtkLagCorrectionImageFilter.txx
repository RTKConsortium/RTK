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

#ifndef __rtkLagCorrectionImageFilter_txx
#define __rtkLagCorrectionImageFilter_txx

//#include "rtkLagCorrectionImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include <iterator>

namespace rtk
{

template<typename TImage, unsigned ModelOrder>
LagCorrectionImageFilter<TImage, ModelOrder>::LagCorrectionImageFilter()
{
	this->SetNumberOfRequiredInputs(1);
	for (unsigned i = 0; i < ModelOrder; ++i) {
		m_A[i] = 0.0f;
		m_B[i] = 0.0f;
	}
	
	m_clock = itkClockType::New();
}

template<typename TImage, unsigned ModelOrder>
void LagCorrectionImageFilter<TImage, ModelOrder>::Initialize()
{
	int i = 0;
	itk::Vector<float, ModelOrder>::Iterator itB = m_B.Begin();
	for (itk::Vector<float, ModelOrder>::Iterator itA = m_A.Begin(); itA != m_A.End(); ++itA, ++itB, ++i) {
		float expma = expf(-*itA);
		m_expma[i] = expma;
		m_bexpma[i] = (*itB) * expma;
	}

	m_ImageId = 0;
}

template<typename TImage, unsigned ModelOrder>
void LagCorrectionImageFilter<TImage, ModelOrder>::ResetInternalState()
{
	//m_S.clear();
	/*if (!m_S->IsNull()) {

	}*/
}

template<typename TImage, unsigned ModelOrder>
void LagCorrectionImageFilter<TImage, ModelOrder>::BeforeThreadedGenerateData()
{
	itkTimeType T1 = m_clock->GetRealTimeStamp().GetTimeInMicroSeconds();
	//ImageRegionType thRegion;
	//m_nThreads = this->SplitRequestedRegion(0, this->GetNumberOfThreads(), thRegion);

	m_nThreads = this->GetNumberOfThreads();
	m_avgCorrection.resize(m_nThreads);
	
	// Initialization at the reception of the first image
	if (!m_ImageId) {
		m_Size = this->GetInput()->GetLargestPossibleRegion().GetSize();
		m_M = m_Size[0] * m_Size[1];

		VectorType v;
		v.Fill(0.0f);
		m_S = StateType::New();
		m_S->SetRegions(this->GetInput()->GetLargestPossibleRegion());
		m_S->Allocate();
		m_S->FillBuffer(v);
	}

	m_thAvgCorr = 0.0f;
	
	std::cout << this->GetInput()->GetLargestPossibleRegion() << std::endl;
	std::cout << this->GetOutput()->GetLargestPossibleRegion() << std::endl;

	itkTimeType T2 = m_clock->GetRealTimeStamp().GetTimeInMicroSeconds();
	std::cout << "Before thread processing: "<<(T2 - T1) << " us" << std::endl;
}

template<typename TImage, unsigned ModelOrder>
void LagCorrectionImageFilter<TImage, ModelOrder>::AfterThreadedGenerateData()
{
	m_avgCorrection.push_back(m_thAvgCorr / (float)m_nThreads);
	std::cout << m_ImageId<< " "<<m_thAvgCorr << " "<< std::endl;
	++m_ImageId;
}

template<typename TImage, unsigned ModelOrder>
void LagCorrectionImageFilter<TImage, ModelOrder>::
ThreadedGenerateData(const ImageRegionType & thRegion, ThreadIdType threadId)
{
	std::cout << "Input" << std::endl;
	itkTimeType T1, T2;
	if (!threadId) {
		T1 = m_clock->GetRealTimeStamp().GetTimeInMicroSeconds();
	}
	// Call for each thread
	
	TImage::Pointer  inputPtr = const_cast< TImage * >(this->GetInput());
	TImage::Pointer outputPtr = this->GetOutput();

	TImage::RegionType reg = thRegion;
	TImage::IndexType start = thRegion.GetIndex();
	start[2] = 0;
	reg.SetIndex(start);

	itk::ImageRegionConstIterator<TImage> itIn(inputPtr, reg);
	itk::ImageRegionIterator<TImage>      itOut(outputPtr, reg);
	itk::ImageRegionIterator<StateType>   itS(m_S, reg);
	
	if (!threadId) {
		std::cout << "--------------------------" << std::endl;
		std::cout << thRegion << std::endl;
		std::cout << reg << std::endl;
		std::cout << this->GetOutput()->GetLargestPossibleRegion() << std::endl;
		std::cout << "--------------------------" << std::endl;
	}

	itIn.GoToBegin();
	itOut.GoToBegin();
	itS.GoToBegin();

	float meanc = 0.0f;     // Average correction over all projection
	float meani = 0.0f;      // Average input image value
	float meanic = 0.0f;     // Average corrected image value
	int rsize = 0;
	while (!itIn.IsAtEnd()) 
	{	
		VectorType S = itS.Get();
				
		// k is pixel id
		float c = 0.0f;
		for (unsigned int n = 0; n<ModelOrder; n++) {
			c += m_bexpma[n] * S[n];
		}
		meanc += c;
		
		meani += itIn.Get();

		PixelType xk = itIn.Get() - (PixelType)c;
		if (xk<0.0f) {
			xk = 0.0f;
		} else if (xk >= 65536) {
			xk = 65535;
		}
		//itIn.Set(xk);    // Check types!
		itOut.Set(xk);    // Check types!

		meanic += xk;

		// Update internal state Snk
		for (unsigned int n = 0; n<ModelOrder; n++) {
			S[n] = (float)xk + m_expma[n] * S[n];
		}

		itS.Set(S);
		
		++itIn;
		++itOut;
		++itS;
		++rsize;
	}
 	meanc = meanc / (float)rsize;
	meani = meani / (float)rsize;
	meanic = meanic / (float)rsize;
	if (!threadId) {
		std::cout << meani << " " << meanic << std::endl;
	}

	m_mutex.Lock();
	m_thAvgCorr += meanc;
	m_mutex.Unlock(); 

	if (!threadId) {
		T2 = m_clock->GetRealTimeStamp().GetTimeInMicroSeconds();
		std::cout << "In-thread processing: " << (T2 - T1) << " us" << std::endl;
	}
}

} // end namespace

#endif
