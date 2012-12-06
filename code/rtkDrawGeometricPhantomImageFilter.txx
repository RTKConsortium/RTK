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
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkDrawCylinderImageFilter.h"
#include "rtkDrawConeImageFilter.h"
#include "itkAddImageFilter.h"

#include "rtkHomogeneousMatrix.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
void DrawGeometricPhantomImageFilter<TInputImage, TOutputImage>::GenerateData()
{
  VectorOfVectorType figParam;
  //Getting phantom parameters
  CFRType::Pointer cfr = CFRType::New();
  cfr->Config(m_ConfigFile);
  figParam = cfr->GetFig();
  VectorType semiprincipalaxis;
  VectorType center;

  //Add Image Filter used to concatenate the different figures obtained on each iteration
  typedef itk::AddImageFilter <TOutputImage, TOutputImage, TOutputImage> AddImageFilterType;
  typename AddImageFilterType::Pointer addFilter = AddImageFilterType::New();

  unsigned int NumberOfFig = figParam.size();
  for(unsigned int i=0; i<NumberOfFig; i++)
  {
    //Set figures parameters
    semiprincipalaxis.push_back(figParam[i][1]);
    semiprincipalaxis.push_back(figParam[i][2]);
    semiprincipalaxis.push_back(figParam[i][3]);
    center.push_back(figParam[i][4]);
    center.push_back(figParam[i][5]);
    center.push_back(figParam[i][6]);

    //Deciding which figure to draw
    switch ((int)figParam[i][0])
    {
      case 0:
      {
        // Create figure object (3D ellipsoid).
        typedef rtk::DrawEllipsoidImageFilter<TInputImage, TOutputImage> DEType;
        typename DEType::Pointer de = DEType::New();
        de->SetInput( this->GetInput() );
        de->SetAxis(semiprincipalaxis);
        de->SetCenter(center);
        de->SetAngle(figParam[i][7]);
        de->SetAttenuation(figParam[i][8]);
        TRY_AND_EXIT_ON_ITK_EXCEPTION( de->Update() );
        addFilter->SetInput1(de->GetOutput());
        if(!i)
          addFilter->SetInput2(this->GetInput());
        else
          addFilter->SetInput2(this->GetOutput());
        break;
      }
      case 1:
      {
        // Create figure object (3D cylinder).
        typedef rtk::DrawCylinderImageFilter<TInputImage, TOutputImage> DCType;
        typename DCType::Pointer dc = DCType::New();
        dc->SetInput( this->GetInput() );
        dc->SetAxis(semiprincipalaxis);
        dc->SetCenter(center);
        dc->SetAngle(figParam[i][7]);
        dc->SetAttenuation(figParam[i][8]);
        TRY_AND_EXIT_ON_ITK_EXCEPTION( dc->Update() );
        addFilter->SetInput1(dc->GetOutput());
        if(!i)
          addFilter->SetInput2(this->GetInput());
        else
          addFilter->SetInput2(this->GetOutput());
        break;
      }
      case 2:
      {
        // Create figure object (3D cone).
        typedef rtk::DrawConeImageFilter<TInputImage, TOutputImage> DCOType;
        typename DCOType::Pointer dco = DCOType::New();
        dco->SetInput( this->GetInput() );
        dco->SetAxis(semiprincipalaxis);
        dco->SetCenter(center);
        dco->SetAngle(figParam[i][7]);
        dco->SetAttenuation(figParam[i][8]);
        TRY_AND_EXIT_ON_ITK_EXCEPTION( dco->Update() );
        addFilter->SetInput1(dco->GetOutput());
        if(!i)
          addFilter->SetInput2(this->GetInput());
        else
          addFilter->SetInput2(this->GetOutput());
        break;
      }
      case 3:
      {
        // Create figure object (3D box).
        typedef rtk::DrawEllipsoidImageFilter<TInputImage, TOutputImage> DBType;
        typename DBType::Pointer db = DBType::New();
        db->SetInput( this->GetInput() );
        db->SetAxis(semiprincipalaxis);
        db->SetCenter(center);
        db->SetAngle(figParam[i][7]);
        db->SetAttenuation(figParam[i][8]);
        TRY_AND_EXIT_ON_ITK_EXCEPTION( db->Update() );
        addFilter->SetInput1(db->GetOutput());
        if(!i)
          addFilter->SetInput2(this->GetInput());
        else
          addFilter->SetInput2(this->GetOutput());
        break;
      }
    }

    TRY_AND_EXIT_ON_ITK_EXCEPTION(addFilter->Update());
    this->GraftOutput( addFilter->GetOutput() );

    semiprincipalaxis.erase(semiprincipalaxis.begin(), semiprincipalaxis.end());
    center.erase(center.begin(), center.end());
  }
}

}// end namespace rtk

#endif
