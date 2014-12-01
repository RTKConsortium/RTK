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

#ifndef __rtkWaterCalibrationImageFilter_h
#define __rtkWaterCalibrationImageFilter_h

#include <vector>
#include <itkImageToImageFilter.h>

#include "rtkConfiguration.h"

namespace rtk
{
/** \class WaterCalibrationImageFilter
 * \brief Performs the weighting for the n-th order of water precorrection
 * (Kachelriess, Med. Phys. 2006)
 *
 * \test rtkwatercalibrationtest.cxx
 *
 * \author S. Brousmiche
 *
 * \ingroup ImageToImageFilter
 */

class ITK_EXPORT WaterCalibrationImageFilter:
  public         itk::ImageToImageFilter< itk::Image< float, 2 >, itk::Image< float, 2 > >
{
public:
  typedef itk::Image< float, 2 > TImage;

  /** Standard class typedefs. */
  typedef WaterCalibrationImageFilter               Self;
  typedef itk::ImageToImageFilter< TImage, TImage > Superclass;
  typedef itk::SmartPointer< Self >                 Pointer;
  typedef itk::SmartPointer< const Self >           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(WaterCalibrationImageFilter, ImageToImageFilter);

  /** Get / Set the Median window that are going to be used during the operation
    */
  itkGetMacro(Order, float);
  itkSetMacro(Order, float);
protected:
  WaterCalibrationImageFilter();
  virtual ~WaterCalibrationImageFilter() {}

  //virtual void BeforeThreadedGenerateData();
  //virtual void AfterThreadedGenerateData();
  virtual void ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId);

private:
  WaterCalibrationImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);         //purposely not implemented

  float m_Order;
};
} // end namespace rtk

#endif
