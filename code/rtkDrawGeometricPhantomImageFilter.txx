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

#ifndef __rtkDrawGeometricPhantomImageFilter_txx
#define __rtkDrawGeometricPhantomImageFilter_txx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkHomogeneousMatrix.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
void DrawGeometricPhantomImageFilter<TInputImage, TOutputImage>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                                                             ThreadIdType itkNotUsed(threadId) )
{
  VectorOfVectorType figParam;
  //Getting phantom parameters
  EQPFunctionType::Pointer sqpFunctor = EQPFunctionType::New();
  //Config File Reader, cfr
  CFRType::Pointer cfr = CFRType::New();
  cfr->Config(m_ConfigFile);
  figParam = cfr->GetFig();
  VectorType semiprincipalaxis;
  VectorType center;

  typename itk::ImageRegionIterator<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);

  typename TOutputImage::PointType point;

  //Iterator at the beginning of the volume
  itOut.GoToBegin();
  unsigned int NumberOfFig = figParam.size();
  for(unsigned int i=0; i<NumberOfFig; i++)
  {
    semiprincipalaxis.push_back(figParam[i][0]);
    semiprincipalaxis.push_back(figParam[i][1]);
    semiprincipalaxis.push_back(figParam[i][2]);
    center.push_back(figParam[i][3]);
    center.push_back(figParam[i][4]);
    center.push_back(figParam[i][5]);
    //Translate from regular expression to quadric
    sqpFunctor->Translate(semiprincipalaxis);
    //Applies rotation and translation if necessary
    sqpFunctor->Rotate(figParam[i][6], center);
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
        itOut.Set(figParam[i][7] + itOut.Get());
      else if (i==0)
        itOut.Set(0.);
      ++itOut;
      }
  semiprincipalaxis.erase(semiprincipalaxis.begin(), semiprincipalaxis.end());
  center.erase(center.begin(), center.end());
  itOut.GoToBegin();
  }
}

}// end namespace rtk

#endif
