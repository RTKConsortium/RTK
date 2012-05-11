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

#ifndef __rtkDrawQuadricImageFilter_h
#define __rtkDrawQuadricImageFilter_h

#include <itkInPlaceImageFilter.h>

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkSetQuadricParamFromRegularParamFunction.h"
#include "rtkConfigFileReader.h"

#include <vector>

namespace rtk
{

/** \class DrawQuadricImageFilter
 * \brief Draws a quadric in an input image.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT DrawQuadricImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef DrawQuadricImageFilter                                    Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage>         Superclass;
  typedef itk::SmartPointer<Self>                                   Pointer;
  typedef itk::SmartPointer<const Self>                             ConstPointer;
  typedef typename TOutputImage::RegionType                         OutputImageRegionType;

  typedef std::vector<double>                                       VectorType;
  typedef std::vector< std::vector<double> >                        VectorOfVectorType;
  typedef std::string                                               StringType;

  typedef rtk::SetQuadricParamFromRegularParamFunction              SQPFunctionType;
  typedef rtk::ConfigFileReader                                     CFRType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawQuadricImageFilter, InPlaceImageFilter);

  /** Get/Set ConfigFile*/
  itkSetMacro(ConfigFile, StringType);
  itkGetMacro(ConfigFile, StringType);

protected:
  DrawQuadricImageFilter() {}
  virtual ~DrawQuadricImageFilter() {};

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );
  /** Translate user parameteres to quadric parameters.
   * A call to this function will assume modification of the function.*/


private:
  DrawQuadricImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented
  StringType m_ConfigFile;

};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawQuadricImageFilter.txx"
#endif

#endif


