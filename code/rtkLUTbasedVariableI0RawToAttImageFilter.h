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

#ifndef __rtkLUTbasedVarI0RawToAttImageFilter_h
#define __rtkLUTbasedVarI0RawToAttImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkVector.h>

#include "rtkConfiguration.h"

namespace rtk
{
/** \class LUTbasedVariableI0RawToAttImageFilter
 * \brief Performs the conversion from raw data to attenuations
 *
 * \test rtklutbasedrawtoattenuationtest.cxx
 *
 * \author S. Brousmiche
 *
 * \ingroup ImageToImageFilter
 */

const unsigned int lutSize = 65536;

class LUTbasedVariableI0RawToAttImageFilter :
  public  itk::ImageToImageFilter< itk::Image< unsigned short, 2 >, itk::Image< float, 2 > >
{
public:
  /** Standard class typedefs. */
  typedef LUTbasedVariableI0RawToAttImageFilter           Self;
  typedef itk::ImageToImageFilter< itk::Image<unsigned short, 2> , itk::Image<float, 2>  > Superclass;
  typedef itk::SmartPointer< Self >                       Pointer;
  typedef itk::SmartPointer< const Self >                 ConstPointer;
  typedef itk::Image< float, 2 >                          OutputImageType;
  typedef OutputImageType::RegionType                     OutputImageRegionType;
  typedef itk::Image< unsigned short, 2 >                 InputImageType;
  typedef itk::Vector< float, lutSize >                   LutType;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(LUTbasedVariableI0RawToAttImageFilter, ImageToImageFilter);

  /** Air level I0
    */
  itkGetMacro(I0, unsigned short);
  itkSetMacro(I0, unsigned short);

protected:
  LUTbasedVariableI0RawToAttImageFilter();
  virtual ~LUTbasedVariableI0RawToAttImageFilter() {}

  virtual void BeforeThreadedGenerateData();
  virtual void ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId);

private:
  LUTbasedVariableI0RawToAttImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);                        //purposely not implemented

  unsigned short m_I0;                                 // Air level I0
  float m_lnI0;                                        // log(I0)
  LutType m_LnILUT;                                    // LUT of ln(I) with I uint16
};
} // end namespace rtk

#endif // __rtkLUTbasedVariableI0RawToAttImageFilter_cxx_
