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

#ifndef __rtkI0EstimationProjectionFilter_h
#define __rtkI0EstimationProjectionFilter_h

#include <itkImageToImageFilter.h>
#include <itkMutexLock.h>
#include <itkBarrier.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConfiguration.h"

#include <vector>
#include <string>

using namespace std;

namespace rtk
{

/** \class I0EstimationProjectionFilter
 * \brief 
 *
 * The output is the image histogram

 * Won't work if saturation
 *
 * \author Sebastien Brousmiche
 *
 * \test rtkI0estimationtest.cxx
 *
 * \ingroup InPlaceImageFilter
 */

template < unsigned char bitShift >
class ITK_EXPORT I0EstimationProjectionFilter : 
	public itk::ImageToImageFilter<itk::Image<unsigned short, 3>, itk::Image<unsigned int, 3> >
{
public:
  /** Standard class typedefs. */
	typedef I0EstimationProjectionFilter<bitShift >            Self;
	typedef itk::ImageToImageFilter<itk::Image<unsigned short, 3>, 
		itk::Image<unsigned, 3> >                                Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

	/** Method for creation through the object factory. */
	itkNewMacro(Self);

	/** Runtime information support. */
	itkTypeMacro(I0EstimationProjectionFilter, ImageToImageFilter);
  
	/** Some convenient typedefs. */
	typedef itk::Image<unsigned int, 3>                 OutputImageType;
	typedef itk::Image<unsigned short, 3>               InputImageType;
	typedef typename OutputImageType::Pointer           OutputImagePointer;
	typedef typename InputImageType::Pointer            InputImagePointer;
	typedef typename InputImageType::ConstPointer       InputImageConstPointer;
	

	// Linear 3-value median filter
	// Is true by default
	itkSetMacro(Median, bool);
	itkGetConstMacro(Median, bool);
	itkBooleanMacro(Median);

	// Is false by default
	itkSetMacro(UseRLS, bool);
	itkGetConstMacro(UseRLS, bool);
	itkBooleanMacro(UseRLS);

	// If RLS set On, uses Lambda=0.7 by default
	itkSetMacro(Lambda, double)
  itkGetMacro(Lambda, double)

	// Use only 1 pixel over 2 when computing histogram
	// Is false by default
	itkSetMacro(UseTurbo, bool);
	itkGetConstMacro(UseTurbo, bool);
	itkBooleanMacro(UseTurbo);
		
	// Expected value from calibration
	itkSetMacro(DebugCSVFile, string)
	itkGetMacro(DebugCSVFile, string)

	// Binding I distance
//	itkSetMacro(BindingDistance, unsigned short)
	//itkGetMacro(BindingDistance, unsigned short)

	// Bind estimated value to expected one
	// Expected I0 must be set accordingly
	itkSetMacro(BindToExpectedI0, bool);
	itkGetConstMacro(BindToExpectedI0, bool);
	itkBooleanMacro(BindToExpectedI0);

	itkGetMacro(I0, unsigned short)   // Estimation result

	itkSetMacro(ExpectedI0, unsigned short)
	itkGetMacro(ExpectedI0, unsigned short)

	//itkGetMacro(I0, unsigned short)
	itkGetMacro(I0fwhm, unsigned short)
	itkGetMacro(I0sigma, float)
	itkGetMacro(I0mean, unsigned short)
	itkGetMacro(I0rls, float)
	itkGetMacro(Np, unsigned int)
	itkGetMacro(Imin, unsigned short)
	itkGetMacro(Imax, unsigned short)
	itkGetMacro(Irange, unsigned short)
	itkGetMacro(highBound, unsigned short)
	itkGetMacro(lowBound, unsigned short)
	itkGetMacro(lowBndRls, unsigned short)
	itkGetMacro(highBndRls, unsigned short)

	
protected:
	I0EstimationProjectionFilter();
	virtual ~I0EstimationProjectionFilter() {}

	//virtual void GenerateOutputInformation();
	//virtual void PropagateRequestedRegion();
	//virtual void GenerateInputRequestedRegion();

  virtual void BeforeThreadedGenerateData();
	
  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId);

	virtual void AfterThreadedGenerateData();

private:
	I0EstimationProjectionFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented
	
	// Input variables 
	bool m_Median;                          // Use median filtering
	bool m_UseRLS;                          // Use RLS filtering
	bool m_UseTurbo;                        // Use turbo (1 pixel/2)
	bool m_BindToExpectedI0;
	unsigned short m_ExpectedI0;            // Expected I0 value (obtained by detector calibration)
	std::string m_DebugCSVFile;             // Save estimates in output CSV file
	double m_Lambda;                        // Forgetting factor for RLS estimate

	unsigned int m_NBins;
	itk::MutexLock::Pointer m_mutex;
	itk::Barrier::Pointer m_Barrier;
	
	unsigned short m_I0;                    // I0 estimate with no a priori
	unsigned short m_I0fwhm;
	float m_I0sigma;
		
	std::vector<unsigned > m_histogram;

	unsigned short m_Imin, m_Imax, m_Irange;

	unsigned short m_I0mean;                // Updated mean estimate (uniform weight over all previous estimates)
	unsigned short m_I0median;              // Median estimate (purpose: increase robustness in case of bad I0 estimate)
	float m_I0rls;                          // RLS estimate
	float m_I0bounded;                      // I0 has maximum in the rls filter IO region
	unsigned int m_Np;                      // Number of previous images

	std::vector<unsigned short> m_pastI0;   // Three past estimate for median filtering
	unsigned short m_lowBound;
	unsigned short m_highBound;
	unsigned short m_middle;
	unsigned short m_lowBndRls, m_highBndRls, m_middleRls;

	float m_dynamicUsage;    // %

	unsigned m_dynThreshold;                  // Minimum number of pixels per bin used in estimation of Imin and Imax

	int m_nsync;
	int m_Nthreads;
}; 

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkI0EstimationProjectionFilter.txx"
#endif

#endif // I0EstimationProjectionFilter.h
