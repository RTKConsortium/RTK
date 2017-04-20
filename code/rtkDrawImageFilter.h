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

#ifndef rtkDrawImageFilter_h
#define rtkDrawImageFilter_h


#include <itkInPlaceImageFilter.h>
#include "rtkDrawSpatialObject.h"
#include "rtkMacro.h"

namespace rtk
{

namespace Functor
{
/**
 * \class Discard
 * \brief
 * \ingroup ITKImageIntensity
 */
template< typename TInput1, typename TInput2 = TInput1, typename TOutput = TInput1 >
class Discard
{
public:

  Discard() {}
  ~Discard() {}
  bool operator!= ( const Discard & ) const
  {
    return false;
  }

  bool operator== ( const Discard & other ) const
  {
    return ! ( *this != other );
  }

  inline TOutput operator() ( const TInput1 & A, const TInput2 itkNotUsed ( &B ) ) const
  {

    return static_cast< TOutput > ( A );
  }
};
}


/** \class DrawImageFilter
 * \brief Base Class for drawing a 3D image by using a DrawSpatialObject. Uses a functor to fill the image.
 *
 * \author Mathieu Dupont
 *
 */

template <class TInputImage,
         class TOutputImage,
         class TSpatialObject,
         typename TFunction
         >
class ITK_EXPORT DrawImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{

public:
  /** Standard class typedefs. */
  typedef DrawImageFilter                                   Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  typedef typename TOutputImage::RegionType                 OutputImageRegionType;
  typedef TSpatialObject                                    SpatialObject;

  /** Method for creation through the object factory. */
  itkNewMacro ( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro ( DrawImageFilter, InPlaceImageFilter );

  itkSetMacro ( Density, double );
  itkGetMacro ( Density, double );

protected:
  DrawImageFilter();
  ~DrawImageFilter() {}
  void ThreadedGenerateData ( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

  TFunction      m_Fillerfunctor;
  TSpatialObject m_SpatialObject;

private:
  DrawImageFilter ( const Self& ); //purposely not implemented
  void operator=(const Self&);     //purposely not implemented
  double         m_Density;

};


} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawImageFilter.hxx"
#endif

#endif
