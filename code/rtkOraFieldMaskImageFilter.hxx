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

#ifndef rtkOraFieldMaskImageFilter_hxx
#define rtkOraFieldMaskImageFilter_hxx

#include <itkImageRegionIteratorWithIndex.h>

#include "rtkHomogeneousMatrix.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"
#include "rtkMacro.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
OraFieldMaskImageFilter<TInputImage,TOutputImage>
::OraFieldMaskImageFilter():
  m_Geometry(ITK_NULLPTR)
{
}

template <class TInputImage, class TOutputImage>
void
OraFieldMaskImageFilter<TInputImage,TOutputImage>
::BeforeThreadedGenerateData()
{
  if(this->GetGeometry()->GetGantryAngles().size() !=
          this->GetOutput()->GetLargestPossibleRegion().GetSize()[2])
    itkExceptionMacro(<<"Number of projections in the input stack and the geometry object differ.")

  // Read meta data dictionary from file
  itk::ImageToImageFilter<TInputImage, TInputImage> *itoi = this;
  itk::ProcessObject *imgsrc = ITK_NULLPTR;
  while(itoi != ITK_NULLPTR)
    {
    imgsrc = itoi->GetInput()->GetSource().GetPointer();
    itoi = dynamic_cast<itk::ImageToImageFilter<TInputImage, TInputImage> *>(imgsrc);
    }
  rtk::ProjectionsReader< TInputImage >* reader;
  reader = dynamic_cast< rtk::ProjectionsReader< TInputImage >* >(imgsrc);
  if(reader == ITK_NULLPTR)
    {
    itkExceptionMacro("Error retrieving rtk::ProjectionsReader");
    }
  itk::ImageIOBase *iofactory;
  iofactory = reader->GetImageIO();
  if (iofactory == ITK_NULLPTR)
    {
    itkExceptionMacro("Error retrieving itk::IOFactory from rtk::ProjectionsReader");
    }
  const itk::MetaDataDictionary &dic = iofactory->GetMetaDataDictionary();

  // Retrieve jaw parameters
  typedef itk::MetaDataObject< double > MetaDataDoubleType;
  const MetaDataDoubleType *meta;
  meta = dynamic_cast<const MetaDataDoubleType *>( dic["xrayx1_cm"]);
  if(meta!=ITK_NULLPTR)
    {
    m_X1 = 10.*meta->GetMetaDataObjectValue();
    }
  else
    {
    itkExceptionMacro("Error reading xrayx1_cm");
    }
  meta = dynamic_cast<const MetaDataDoubleType *>( dic["xrayx2_cm"]);
  if(meta!=ITK_NULLPTR)
    {
    m_X2 = 10.*meta->GetMetaDataObjectValue();
    }
  else
    {
    itkExceptionMacro("Error reading xrayx2_cm");
    }
  meta = dynamic_cast<const MetaDataDoubleType *>( dic["xrayy1_cm"]);
  if(meta!=ITK_NULLPTR)
    {
    m_Y1 = 10.*meta->GetMetaDataObjectValue();
    }
  else
    {
    itkExceptionMacro("Error reading xrayy1_cm");
    }
  meta = dynamic_cast<const MetaDataDoubleType *>( dic["xrayy2_cm"]);
  if(meta!=ITK_NULLPTR)
    {
    m_Y2 = 10.*meta->GetMetaDataObjectValue();
    }
  else
    {
    itkExceptionMacro("Error reading xrayy2_cm");
    }
}

template <class TInputImage, class TOutputImage>
void
OraFieldMaskImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType itkNotUsed(threadId) )
{
  // Iterators on input and output
  typedef ProjectionsRegionConstIteratorRayBased<TInputImage> InputRegionIterator;
  InputRegionIterator *itIn;
  itIn = InputRegionIterator::New(this->GetInput(),
                                  outputRegionForThread,
                                  m_Geometry);
  typedef itk::ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Go over each projection
  for(unsigned int pix=0; pix<outputRegionForThread.GetNumberOfPixels(); pix++, itIn->Next(), ++itOut)
    {
    typedef typename InputRegionIterator::PointType PointType;
    PointType source = itIn->GetSourcePosition();
    double sourceNorm = source.GetNorm();
    PointType pixel = itIn->GetPixelPosition();
    double sourcey = source[1];
    pixel[1] -= sourcey;
    source[1] = 0.;
    PointType pixelToSource(source-pixel);
    PointType sourceDir = source/sourceNorm;

    // Get the 3D position of the ray and the main plane
    // (at 0,0,0 and orthogonal to source to center)
    const double mag = sourceNorm/(pixelToSource*sourceDir);
    PointType intersection = source - mag*pixelToSource;

    PointType v(0.);
    v[1] = 1.;

    PointType u = CrossProduct(v, sourceDir);
    double ucoord = u*intersection;
    double vcoord = intersection[1]; // equivalent to v*intersection
    if( ucoord > -1.*m_X1 &&
        ucoord <     m_X2 &&
        vcoord > -1.*m_Y1 &&
        vcoord <     m_Y2 )
      {
      itOut.Set( itIn->Get() );
      }
    else
      {
      itOut.Set( 0. );
      }
    }

  delete itIn;
}

} // end namespace rtk

#endif
