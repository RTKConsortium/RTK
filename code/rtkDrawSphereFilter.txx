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

#ifndef __rtkDrawSphereFilter_txx
#define __rtkDrawSphereFilter_txx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkHomogeneousMatrix.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
DrawSphereFilter<TInputImage, TOutputImage>
::DrawSphereFilter():
m_SphereScale(128.0)
{
}

template <class TInputImage, class TOutputImage>
void DrawSphereFilter<TInputImage, TOutputImage>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                                                             ThreadIdType itkNotUsed(threadId) )
{
  //Getting phantom parameters
  EQPFunctionType::Pointer sqpFunctor = EQPFunctionType::New();
  figure figParam;
  std::vector< figure > shepplogan(1, figParam);

  shepplogan[0].semiprincipalaxis.push_back(0.69*m_SphereScale);
  shepplogan[0].semiprincipalaxis.push_back(0.90*m_SphereScale);
  shepplogan[0].semiprincipalaxis.push_back(0.92*m_SphereScale);
  shepplogan[0].center.push_back(0.);
  shepplogan[0].center.push_back(0.);
  shepplogan[0].center.push_back(0.);
  shepplogan[0].angle = 0.;
  shepplogan[0].attenuation = 1.;

  typename TOutputImage::PointType point;
  const    TInputImage *           input = this->GetInput();

    typename itk::ImageRegionConstIterator<TInputImage> itIn( input, outputRegionForThread);
    typename itk::ImageRegionIterator<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);

    //Translate from regular expression to quadric
    sqpFunctor->Translate(shepplogan[0].semiprincipalaxis);
    //Applies rotation and translation if necessary
    sqpFunctor->Rotate(shepplogan[0].angle, shepplogan[0].center);
    while( !itOut.IsAtEnd() )
      {
      this->GetInput()->TransformIndexToPhysicalPoint(itOut.GetIndex(), point);

      double QuadricEllip = sqpFunctor->GetA()*point[0]*point[0]   +
                   sqpFunctor->GetB()*point[1]*point[1]   +
                   sqpFunctor->GetC()*point[2]*point[2]   +
                   sqpFunctor->GetD()*point[0]*point[1]   +
                   sqpFunctor->GetE()*point[0]*point[2]   +
                   sqpFunctor->GetF()*point[1]*point[2]   +
                   sqpFunctor->GetG()*point[0] + sqpFunctor->GetH()*point[1] +
                   sqpFunctor->GetI()*point[2] + sqpFunctor->GetJ();
      if(QuadricEllip<0)
        itOut.Set(shepplogan[0].attenuation + itIn.Get());
      else
        itOut.Set(0.);
      ++itIn;
      ++itOut;
      }
    input = this->GetOutput();
}

}// end namespace rtk

#endif
