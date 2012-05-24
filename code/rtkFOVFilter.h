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

#ifndef __rtkFOVFilter_h
#define __rtkFOVFilter_h

#include <itkImageToImageFilter.h>

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkSetQuadricParamFromRegularParamFunction.h"
#include "rtkConfigFileReader.h"

#include <vector>

namespace rtk
{

/** \class FOVFilter
 * \brief Masks out the regions that are not included in our FOV.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT FOVFilter :
  public itk::ImageToImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FOVFilter                                                 Self;
  typedef itk::ImageToImageFilter<TInputImage,TOutputImage>         Superclass;
  typedef itk::SmartPointer<Self>                                   Pointer;
  typedef itk::SmartPointer<const Self>                             ConstPointer;
  typedef typename TOutputImage::RegionType                         OutputImageRegionType;
  typedef rtk::ThreeDCircularProjectionGeometry                     GeometryType;
  typedef typename GeometryType::Pointer                            GeometryPointer;


  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FOVFilter, ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

  /** Get / Set of the member FOVradius */
  itkGetMacro(FOVradius, double);
  itkSetMacro(FOVradius, double);

  /** Get / Set of the member ChineseHatHeight */
  itkGetMacro(ChineseHatHeight, double);
  itkSetMacro(ChineseHatHeight, double);

  /** Get / Set of the member Mask */
  itkGetMacro(Mask, bool);
  itkSetMacro(Mask, bool);

protected:
  FOVFilter();
  virtual ~FOVFilter() {};
  /** Computes the radius of the FOV cylinder */
  void ComputationFOVradius();
  /** Computes the height of the chinese hats present in the FOV */
  void ComputationChineseHatHeight();
  /** Generates a FOV mask which is applied to the reconstruction
   * A call to this function will assume modification of the function.*/
  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

private:
  FOVFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented
  double                    m_FOVradius;
  double                    m_ChineseHatHeight;
  GeometryPointer           m_Geometry;
  bool                      m_Mask;

};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFOVFilter.txx"
#endif

#endif


