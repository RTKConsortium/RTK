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

#ifndef __rtkDrawSheppLoganFilter_txx
#define __rtkDrawSheppLoganFilter_txx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include "rtkHomogeneousMatrix.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
void DrawSheppLoganFilter<TInputImage, TOutputImage>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                                                             ThreadIdType threadId )
{
  //Getting phantom parameters
  SQPFunctionType::Pointer sqpFunctor = SQPFunctionType::New();
  figure figParam;
  std::vector< figure > shepplogan(10, figParam);
  unsigned int NumberOfFig = 10;

  shepplogan[0].semiprincipalaxis.push_back(88.32*0.7);
  shepplogan[0].semiprincipalaxis.push_back(117.76*0.7);
  shepplogan[0].semiprincipalaxis.push_back(115.6*0.7);
  shepplogan[0].center.push_back(0.);
  shepplogan[0].center.push_back(0.);
  shepplogan[0].center.push_back(0.);
  shepplogan[0].angle = 0.;
  shepplogan[0].attenuation = 2.;

  shepplogan[1].semiprincipalaxis.push_back(84.7872*0.7);
  shepplogan[1].semiprincipalaxis.push_back(111.872*0.7);
  shepplogan[1].semiprincipalaxis.push_back(112.64*0.7);
  shepplogan[1].center.push_back(0.);
  shepplogan[1].center.push_back(0.);
  shepplogan[1].center.push_back(0.);
  shepplogan[1].angle = 0.;
  shepplogan[1].attenuation = -0.98;

  shepplogan[2].semiprincipalaxis.push_back(52.48*0.7);
  shepplogan[2].semiprincipalaxis.push_back(20.48*0.7);
  shepplogan[2].semiprincipalaxis.push_back(26.88*0.7);
  shepplogan[2].center.push_back(-28.16*0.7);
  shepplogan[2].center.push_back(0.);
  shepplogan[2].center.push_back(-32.*0.7);
  shepplogan[2].angle = 108.;
  shepplogan[2].attenuation = -0.02;

  shepplogan[3].semiprincipalaxis.push_back(39.68*0.7);
  shepplogan[3].semiprincipalaxis.push_back(14.08*0.7);
  shepplogan[3].semiprincipalaxis.push_back(28.16*0.7);
  shepplogan[3].center.push_back(28.16*0.7);
  shepplogan[3].center.push_back(0.);
  shepplogan[3].center.push_back(-32*0.7);
  shepplogan[3].angle = 72.;
  shepplogan[3].attenuation = -0.02;

  shepplogan[4].semiprincipalaxis.push_back(26.88*0.7);
  shepplogan[4].semiprincipalaxis.push_back(32*0.7);
  shepplogan[4].semiprincipalaxis.push_back(64*0.7);
  shepplogan[4].center.push_back(0.);
  shepplogan[4].center.push_back(-44.8*0.7);
  shepplogan[4].center.push_back(-32*0.7);
  shepplogan[4].angle = 0.;
  shepplogan[4].attenuation = 0.02;

  shepplogan[5].semiprincipalaxis.push_back(5.888*0.7);
  shepplogan[5].semiprincipalaxis.push_back(5.888*0.7);
  shepplogan[5].semiprincipalaxis.push_back(5.888*0.7);
  shepplogan[5].center.push_back(0.);
  shepplogan[5].center.push_back(-12.8*0.7);
  shepplogan[5].center.push_back(-32*0.7);
  shepplogan[5].angle = 0.;
  shepplogan[5].attenuation = 0.02;

  shepplogan[6].semiprincipalaxis.push_back(5.888*0.7);
  shepplogan[6].semiprincipalaxis.push_back(2.944*0.7);
  shepplogan[6].semiprincipalaxis.push_back(2.56*0.7);
  shepplogan[6].center.push_back(-10.24*0.7);
  shepplogan[6].center.push_back(83.2*0.7);
  shepplogan[6].center.push_back(-32*0.7);
  shepplogan[6].angle = 0.;
  shepplogan[6].attenuation = 0.01;

  shepplogan[7].semiprincipalaxis.push_back(5.888*0.7);
  shepplogan[7].semiprincipalaxis.push_back(2.944*0.7);
  shepplogan[7].semiprincipalaxis.push_back(2.56*0.7);
  shepplogan[7].center.push_back(7.68*0.7);
  shepplogan[7].center.push_back(83.2*0.7);
  shepplogan[7].center.push_back(-32*0.7);
  shepplogan[7].angle = 90.;
  shepplogan[7].attenuation = 0.01;

  shepplogan[8].semiprincipalaxis.push_back(7.168*0.7);
  shepplogan[8].semiprincipalaxis.push_back(5.12*0.7);
  shepplogan[8].semiprincipalaxis.push_back(12.8*0.7);
  shepplogan[8].center.push_back(7.68*0.7);
  shepplogan[8].center.push_back(13.44*0.7);
  shepplogan[8].center.push_back(80.*0.7);
  shepplogan[8].angle = 90.;
  shepplogan[8].attenuation = 0.02;

  shepplogan[9].semiprincipalaxis.push_back(7.168*0.7);
  shepplogan[9].semiprincipalaxis.push_back(7.168*0.7);
  shepplogan[9].semiprincipalaxis.push_back(12.8*0.7);
  shepplogan[9].center.push_back(0.);
  shepplogan[9].center.push_back(-12.8*0.7);
  shepplogan[9].center.push_back(80.*0.7);
  shepplogan[9].angle = 0.;
  shepplogan[9].attenuation = -0.02;

  typename itk::ImageRegionIterator<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);

  typename TOutputImage::PointType point;

  //Iterator at the beginning of the volume
  itOut.GoToBegin();

  for(unsigned int i=0; i<NumberOfFig; i++)
  {
    //Translate from regular expression to quadric
    sqpFunctor->Translate(shepplogan[i].semiprincipalaxis);
    //Applies rotation and translation if necessary
    sqpFunctor->Rotate(shepplogan[i].angle, shepplogan[i].center);
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
        itOut.Set(shepplogan[i].attenuation + itOut.Get());
      else if (i==0)
        itOut.Set(0.);
      ++itOut;
      }
  itOut.GoToBegin();
  }
}

}// end namespace rtk

#endif
