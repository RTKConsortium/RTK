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

#ifndef rtkDisplacedDetectorForOffsetFieldOfViewImageFilter_h
#define rtkDisplacedDetectorForOffsetFieldOfViewImageFilter_h

#include "rtkDisplacedDetectorImageFilter.h"

namespace rtk
{

/** \class DisplacedDetectorForOffsetFieldOfViewImageFilter
 * \brief Weigting for displaced detectors with offset field-of-view
 *
 * The class does something similar to rtk::DisplacedDetectorImageFilter but
 * handles in addition the case of a field-of-view that is not centered on the
 * center of rotation.
 *
 * \test rtkdisplaceddetectorcompoffsettest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT DisplacedDetectorForOffsetFieldOfViewImageFilter :
  public rtk::DisplacedDetectorImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef DisplacedDetectorForOffsetFieldOfViewImageFilter             Self;
  typedef rtk::DisplacedDetectorImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                      Pointer;
  typedef itk::SmartPointer<const Self>                                ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                                     InputImageType;
  typedef TOutputImage                                    OutputImageType;
  typedef typename OutputImageType::RegionType            OutputImageRegionType;
  typedef itk::Image<typename TOutputImage::PixelType, 1> WeightImageType;

  typedef ThreeDCircularProjectionGeometry GeometryType;
  typedef GeometryType::Pointer            GeometryPointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(DisplacedDetectorForOffsetFieldOfViewImageFilter, ImageToImageFilter);

protected:
  DisplacedDetectorForOffsetFieldOfViewImageFilter();
  ~DisplacedDetectorForOffsetFieldOfViewImageFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;

  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

private:
  DisplacedDetectorForOffsetFieldOfViewImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);               //purposely not implemented

  /**
   * Center coordinates and size of the FOV cylinder.
   */
  double m_FOVRadius;
  double m_FOVCenterX;
  double m_FOVCenterZ;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDisplacedDetectorForOffsetFieldOfViewImageFilter.hxx"
#endif

#endif
