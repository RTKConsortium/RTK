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
	public itk::ImageToImageFilter<itk::Image<unsigned short, 3>, itk::Image<unsigned short, 3> >
{
public:
  /** Standard class typedefs. */
	typedef I0EstimationProjectionFilter<bitShift >            Self;
	typedef itk::ImageToImageFilter<itk::Image<unsigned short, 3>, 
		itk::Image<unsigned int, 3> >                          Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
	typedef itk::Image<unsigned short, 3>                      ImageType;
	typedef itk::Image<unsigned short, 3>                      HistogramType;
  
  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
	itkTypeMacro(I0EstimationProjectionFilter, itk::ImageToImageFilter);


	// Is true by default
	itkSetMacro(Median, bool);
	itkGetConstMacro(Median, bool);
	itkBooleanMacro(Median);

	// Is false by default
	itkSetMacro(UseRLS, bool);
	itkGetConstMacro(UseRLS, bool);
	itkBooleanMacro(UseRLS);

	// Use only 1 pixel over 2 when computing histogram
	// Is false by default
	itkSetMacro(UseTurbo, bool);
	itkGetConstMacro(UseTurbo, bool);
	itkBooleanMacro(UseTurbo);
	
	// Expected value from calibration
	itkSetMacro(expectedI0, unsigned short)
	itkGetMacro(expectedI0, unsigned short)

	// Binding I distance
//	itkSetMacro(BindingDistance, unsigned short)
	//itkGetMacro(BindingDistance, unsigned short)

	// Bind estimated value to expected one
	// Expected I0 must be set accordingly
//	itkSetMacro(BindToExpectedI0, bool);
//	itkGetConstMacro(BindToExpectedI0, bool);
//	itkBooleanMacro(BindToExpectedI0);



	itkGetMacro(I0, unsigned short)   // Estimation result

	itkSetMacro(Lambda, double)
	itkGetMacro(Lambda, double)

	//itkGetMacro(I0, unsigned short)
	itkGetMacro(I0fwhm, unsigned short)
	itkGetMacro(I0mean, double)
	itkGetMacro(I0rls, double)
	itkGetMacro(Np, unsigned int)
	
protected:
	I0EstimationProjectionFilter();
	virtual ~I0EstimationProjectionFilter() {}

  virtual void BeforeThreadedGenerateData();
	
  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId);

	virtual void AfterThreadedGenerateData();

private:
	I0EstimationProjectionFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented
	
	unsigned int m_NBins;
	itk::MutexLock::Pointer m_mutex;
	itk::Barrier::Pointer m_Barrier;

	bool m_Median;
	bool m_UseRLS;
	bool m_UseTurbo;
	unsigned short m_expectedI0;

	unsigned short m_I0;
	unsigned short m_I0fwhm;
	
	
	std::vector<unsigned > m_histogram;

	unsigned m_Imin, m_Imax, m_Irange;
	double m_I0mean;                        // Updated mean estimate (uniform weight over all previous estimates)
	float m_Lambda;                         // Forgetting factor for RLS estimate
	double m_I0rls;                         // RLS estimate
	unsigned int m_Np;                      // Number of previous images

	std::vector<unsigned short> m_pastI0;   // Three past estimate for median filtering
	unsigned short m_lowBound;
	unsigned short m_highBound;
	unsigned short m_middle;

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
