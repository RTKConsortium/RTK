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

#ifndef rtkDrawSheppLoganFilter_hxx
#define rtkDrawSheppLoganFilter_hxx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkHomogeneousMatrix.h"

#include "rtkMacro.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
DrawSheppLoganFilter<TInputImage, TOutputImage>
::DrawSheppLoganFilter():
m_PhantomScale(128.0),
m_OriginOffset(0.)
{
}

template< class TInputImage, class TOutputImage >
void DrawSheppLoganFilter<TInputImage, TOutputImage>::SetEllipsoid( FigureType* ellipsoid,
                                                                    double spax,
                                                                    double spay,
                                                                    double spaz,
                                                                    double centerx,
                                                                    double centery,
                                                                    double centerz,
                                                                    double angle,
                                                                    double density)
{
  ellipsoid->semiprincipalaxis[0] = spax * m_PhantomScale[0];
  ellipsoid->semiprincipalaxis[1] = spay * m_PhantomScale[1];
  ellipsoid->semiprincipalaxis[2] = spaz * m_PhantomScale[2];
  ellipsoid->center[0] = (centerx + m_OriginOffset[0]) * m_PhantomScale[0];
  ellipsoid->center[1] = (centery + m_OriginOffset[1]) * m_PhantomScale[1];
  ellipsoid->center[2] = (centerz + m_OriginOffset[2]) * m_PhantomScale[2];
  ellipsoid->angle = angle;
  ellipsoid->attenuation = density;
}

template <class TInputImage, class TOutputImage>
void DrawSheppLoganFilter<TInputImage, TOutputImage>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                                                             ThreadIdType itkNotUsed(threadId) )
{
  //Getting phantom parameters
  EQPFunctionType::Pointer sqpFunctor = EQPFunctionType::New();
  FigureType figParam;
  std::vector< FigureType > shepplogan(10, figParam);
  unsigned int NumberOfFig = 10;

  SetEllipsoid(&(shepplogan[0]), 0.69, 0.90, 0.92, 0., 0., 0., 0., 2.);
  SetEllipsoid(&(shepplogan[1]), 0.6624, 0.880, 0.874, 0., 0., 0., 0., -0.98);
  SetEllipsoid(&(shepplogan[2]), 0.41, 0.21, 0.16, -0.22, -0.25, 0., 108., -0.02);
  SetEllipsoid(&(shepplogan[3]), 0.31, 0.22, 0.11, 0.22, -0.25, 0., 72., -0.02);
  SetEllipsoid(&(shepplogan[4]), 0.21, 0.50, 0.25, 0., -0.25, 0.35, 0., 0.02);
  SetEllipsoid(&(shepplogan[5]), 0.046, 0.046, 0.046, 0., -0.25, 0.10, 0., 0.02);
  SetEllipsoid(&(shepplogan[6]), 0.046, 0.02, 0.023, -0.08, -0.25, -0.65, 0., 0.01);
  SetEllipsoid(&(shepplogan[7]), 0.046, 0.02, 0.023, 0.06, -0.25, -0.65, 90., 0.01);
  SetEllipsoid(&(shepplogan[8]), 0.056, 0.01, 0.04, 0.06, 0.625, -0.105, 90., 0.02);
  SetEllipsoid(&(shepplogan[9]), 0.056, 0.1, 0.056, 0., 0.625, 0.1, 0., -0.02);

  typename TOutputImage::PointType point;
  const    TInputImage *           input = this->GetInput();

  for(unsigned int i=0; i<NumberOfFig; i++)
  {
    typename itk::ImageRegionConstIterator<TInputImage> itIn( input, outputRegionForThread);
    typename itk::ImageRegionIterator<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);

    //Set type of Figure
    sqpFunctor->SetFigure("Ellipsoid");

    //Translate from regular expression to quadric
    sqpFunctor->Translate(shepplogan[i].semiprincipalaxis);

    //Applies rotation and translation if necessary
    sqpFunctor->Rotate(shepplogan[i].angle, shepplogan[i].center);

    while( !itOut.IsAtEnd() )
      {
      this->GetInput()->TransformIndexToPhysicalPoint(itOut.GetIndex(), point);

      double QuadricEllip =
                   sqpFunctor->GetA()*point[0]*point[0]   +
                   sqpFunctor->GetB()*point[1]*point[1]   +
                   sqpFunctor->GetC()*point[2]*point[2]   +
                   sqpFunctor->GetD()*point[0]*point[1]   +
                   sqpFunctor->GetE()*point[0]*point[2]   +
                   sqpFunctor->GetF()*point[1]*point[2]   +
                   sqpFunctor->GetG()*point[0] + sqpFunctor->GetH()*point[1] +
                   sqpFunctor->GetI()*point[2] + sqpFunctor->GetJ();
      if(QuadricEllip<0){
        itOut.Set(shepplogan[i].attenuation + itIn.Get());
      }
      else if (i==0)
        itOut.Set(0.);
      ++itIn;
      ++itOut;
      }
    input = this->GetOutput();
  }
}

}// end namespace rtk

#endif
