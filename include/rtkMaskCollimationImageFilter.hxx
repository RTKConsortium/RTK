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

#ifndef rtkMaskCollimationImageFilter_hxx
#define rtkMaskCollimationImageFilter_hxx

#include <itkImageRegionIteratorWithIndex.h>

#include "rtkHomogeneousMatrix.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"
#include "rtkMacro.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
MaskCollimationImageFilter<TInputImage,TOutputImage>
::MaskCollimationImageFilter():
  m_Geometry(ITK_NULLPTR)
{
}

template <class TInputImage, class TOutputImage>
void
MaskCollimationImageFilter<TInputImage,TOutputImage>
::BeforeThreadedGenerateData()
{
  if(this->GetGeometry()->GetGantryAngles().size() !=
          this->GetOutput()->GetLargestPossibleRegion().GetSize()[2])
    itkExceptionMacro(<<"Number of projections in the input stack and the geometry object differ.")
}

template <class TInputImage, class TOutputImage>
void
MaskCollimationImageFilter<TInputImage,TOutputImage>
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
  unsigned int npixperslice;
  npixperslice = outputRegionForThread.GetSize(0) *
                 outputRegionForThread.GetSize(1);
  typedef typename OutputImageRegionType::SizeValueType SizeValueType;
  for(SizeValueType iProj=outputRegionForThread.GetIndex(2);
                    iProj<outputRegionForThread.GetIndex(2)+
                          outputRegionForThread.GetSize(2);
                    iProj++)
    {
    for(unsigned int pix=0;
                     pix<npixperslice;
                     pix++, itIn->Next(), ++itOut)
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
      if( ucoord > -1.*this->GetGeometry()->GetCollimationUInf()[iProj] &&
          ucoord <     this->GetGeometry()->GetCollimationUSup()[iProj] &&
          vcoord > -1.*this->GetGeometry()->GetCollimationVInf()[iProj] &&
          vcoord <     this->GetGeometry()->GetCollimationVSup()[iProj] )
        {
        itOut.Set( itIn->Get() );
        }
      else
        {
        itOut.Set( 0. );
        }
      }
    }

  delete itIn;
}

} // end namespace rtk

#endif
