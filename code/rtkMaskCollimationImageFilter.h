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

#ifndef rtkMaskCollimationImageFilter_h
#define rtkMaskCollimationImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkConfiguration.h"
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

/** \class MaskCollimationImageFilter
 * \brief Mask out everything behind the jaws (typically in Ora file format,
 * i.e., the medPhoton scanner)
 *
 * \test rtkoratest
 *
 * \author Simon Rit
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT MaskCollimationImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef MaskCollimationImageFilter                        Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  typedef typename TOutputImage::RegionType               OutputImageRegionType;
  typedef rtk::ThreeDCircularProjectionGeometry           GeometryType;
  typedef typename GeometryType::Pointer                  GeometryPointer;
  typedef std::vector<std::string>                        FileNamesContainer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MaskCollimationImageFilter, itk::ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

protected:
  MaskCollimationImageFilter();
  ~MaskCollimationImageFilter() {};

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  /** Apply changes to the input image requested region. */
  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread,
                             ThreadIdType threadId ) ITK_OVERRIDE;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

private:
  MaskCollimationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);          //purposely not implemented

  /** RTK geometry object */
  GeometryPointer m_Geometry;

  /** Jaw positions */
  double m_X1;
  double m_X2;
  double m_Y1;
  double m_Y2;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkMaskCollimationImageFilter.hxx"
#endif

#endif
