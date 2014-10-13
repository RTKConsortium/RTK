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
		itk::Image<unsigned int, 3> >                          Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
	typedef itk::Image<unsigned short, 3>                      ImageType;
	typedef itk::Image<unsigned int, 3>                        HistogramType;
  
  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
	itkTypeMacro(I0EstimationProjectionFilter, itk::ImageToImageFilter);
	
protected:
	I0EstimationProjectionFilter();
	virtual ~I0EstimationProjectionFilter() {}

	//virtual void GenerateOutputInformation(); // Create output instogram.

  virtual void BeforeThreadedGenerateData();
	
  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId);

	virtual void AfterThreadedGenerateData();

private:
	I0EstimationProjectionFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented
	
	unsigned int m_NBins;
	itk::MutexLock::Pointer m_mutex;

	std::vector<unsigned > m_histogram;

}; 

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkI0EstimationProjectionFilter.txx"
#endif

#endif // I0EstimationProjectionFilter.h
