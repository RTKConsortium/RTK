#ifndef __itkDrawQuadricFunctor_txx
#define __itkDrawQuadricFunctor_txx

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

namespace itk
{

template <class TInputImage, class TOutputImage>
void DrawQuadricFunctor<TInputImage, TOutputImage>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                                                                   ThreadIdType threadId )
{
  std::vector< std::vector<double> > Fig;
  //Getting phantom parameters
  m_SQPFunctor = SQPFunctionType::New();
  m_SQPFunctor->Config(m_ConfigFile);
  Fig = m_SQPFunctor->GetFig();
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
    m_SQPFunctor->Translate(semiprincipalaxis);
    //Applies rotation and translation if necessary
    m_SQPFunctor->Rotate(Fig[i][6], center);
    while( !itOut.IsAtEnd() )
    {
    this->GetInput()->TransformIndexToPhysicalPoint(itOut.GetIndex(), point);

    double QuadricEllip = m_SQPFunctor->GetA()*point[0]*point[0]   +
                 m_SQPFunctor->GetB()*point[1]*point[1]   +
                 m_SQPFunctor->GetC()*point[2]*point[2]   +
                 m_SQPFunctor->GetD()*point[0]*point[1]   +
                 m_SQPFunctor->GetE()*point[0]*point[2]   +
                 m_SQPFunctor->GetF()*point[1]*point[2]   +
                 m_SQPFunctor->GetG()*point[0] + m_SQPFunctor->GetH()*point[1] +
                 m_SQPFunctor->GetI()*point[2] + m_SQPFunctor->GetJ();
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
}// end namespace itk

#endif
