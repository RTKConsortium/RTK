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

#ifndef __rtkDrawQuadricFunctor_txx
#define __rtkDrawQuadricFunctor_txx

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include "rtkHomogeneousMatrix.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
void DrawQuadricFunctor<TInputImage, TOutputImage>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                                                                   ThreadIdType threadId )
{
  std::vector< std::vector<double> > Fig;
  //Getting phantom parameters
  SQPFunctionType::Pointer sqpFunctor = SQPFunctionType::New();
  sqpFunctor->Config(m_ConfigFile);
  Fig = sqpFunctor->GetFig();
  VectorType semiprincipalaxis;
  VectorType center;
  //Creating 3D Image and Point
  typedef itk::Image< OutputPixelType, 3 >  OutputImageType;
  itk::ImageRegionIterator<OutputImageType> itOut(this->GetOutput(), outputRegionForThread);

  typename OutputImageType::PointType point;

  //Iterator at the beginning of the volume
  itOut.GoToBegin();
  unsigned int NumberOfFig = Fig.size();
  for(unsigned int i=0; i<NumberOfFig; i++)
  {
    semiprincipalaxis.push_back(Fig[i][0]);
    semiprincipalaxis.push_back(Fig[i][1]);
    semiprincipalaxis.push_back(Fig[i][2]);
    center.push_back(Fig[i][3]);
    center.push_back(Fig[i][4]);
    center.push_back(Fig[i][5]);
    //Translate from regular expression to quadric
    sqpFunctor->Translate(semiprincipalaxis);
    //Applies rotation and translation if necessary
    sqpFunctor->Rotate(Fig[i][6], center);
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
      itOut.Set(Fig[i][7] + itOut.Get());
    else
    {
      if (i==0)
        itOut.Set(0.);
    }
    ++itOut;
    }
  semiprincipalaxis.erase(semiprincipalaxis.begin(), semiprincipalaxis.end());
  center.erase(center.begin(), center.end());
  itOut.GoToBegin();
  }
}
}// end namespace rtk

#endif
