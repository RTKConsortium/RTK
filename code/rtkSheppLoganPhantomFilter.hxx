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

#ifndef __rtkSheppLoganPhantomFilter_hxx
#define __rtkSheppLoganPhantomFilter_hxx


namespace rtk
{
template <class TInputImage, class TOutputImage>
SheppLoganPhantomFilter<TInputImage, TOutputImage>
::SheppLoganPhantomFilter():
 m_PhantomScale(128.),
 m_OriginOffset(0.)
{
}

template< class TInputImage, class TOutputImage >
void SheppLoganPhantomFilter< TInputImage, TOutputImage >::GenerateData()
{
  std::vector< typename REIType::Pointer > rei(10);
  unsigned int NumberOfFig = 10;
  for ( unsigned int j = 0; j < NumberOfFig; j++ )
    {
      rei[j] = REIType::New();
    }

  VectorType semiprincipalaxis, center;
  semiprincipalaxis[0] = 0.69;
  semiprincipalaxis[1] = 0.90;
  semiprincipalaxis[2] = 0.92;
  center[0] = 0.;
  center[1] = 0.;
  center[2] = 0.;
  rei[0]->SetAxis(m_PhantomScale * semiprincipalaxis);
  rei[0]->SetCenter(m_PhantomScale * (m_OriginOffset + center) );
  rei[0]->SetAngle(0.);
  rei[0]->SetDensity(2.);
  rei[0]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.6624;
  semiprincipalaxis[1] = 0.880;
  semiprincipalaxis[2] = 0.874;
  center[0] = 0.;
  center[1] = 0.;
  center[2] = 0.;
  rei[1]->SetAxis(m_PhantomScale * semiprincipalaxis);
  rei[1]->SetCenter(m_PhantomScale * (m_OriginOffset + center) );
  rei[1]->SetAngle(0.);
  rei[1]->SetDensity(-0.98);
  rei[1]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.41;
  semiprincipalaxis[1] = 0.21;
  semiprincipalaxis[2] = 0.16;
  center[0] = -0.22;
  center[1] = -0.25;
  center[2] = 0.;
  rei[2]->SetAxis(m_PhantomScale * semiprincipalaxis);
  rei[2]->SetCenter(m_PhantomScale * (m_OriginOffset + center) );
  rei[2]->SetAngle(108.);
  rei[2]->SetDensity(-0.02);
  rei[2]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.31;
  semiprincipalaxis[1] = 0.22;
  semiprincipalaxis[2] = 0.11;
  center[0] = 0.22;
  center[1] = -0.25;
  center[2] = 0.;
  rei[3]->SetAxis(m_PhantomScale * semiprincipalaxis);
  rei[3]->SetCenter(m_PhantomScale * (m_OriginOffset + center) );
  rei[3]->SetAngle(72.);
  rei[3]->SetDensity(-0.02);
  rei[3]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.21;
  semiprincipalaxis[1] = 0.50;
  semiprincipalaxis[2] = 0.25;
  center[0] = 0.;
  center[1] = -0.25;
  center[2] = 0.35;
  rei[4]->SetAxis(m_PhantomScale * semiprincipalaxis);
  rei[4]->SetCenter(m_PhantomScale * (m_OriginOffset + center) );
  rei[4]->SetAngle(0.);
  rei[4]->SetDensity(0.02);
  rei[4]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.046;
  semiprincipalaxis[1] = 0.046;
  semiprincipalaxis[2] = 0.046;
  center[0] = 0.;
  center[1] = -0.25;
  center[2] = 0.10;
  rei[5]->SetAxis(m_PhantomScale * semiprincipalaxis);
  rei[5]->SetCenter(m_PhantomScale * (m_OriginOffset + center) );
  rei[5]->SetAngle(0.);
  rei[5]->SetDensity(0.02);
  rei[5]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.046;
  semiprincipalaxis[1] = 0.020;
  semiprincipalaxis[2] = 0.023;
  center[0] = -0.08;
  center[1] = -0.250;
  center[2] = -0.650;
  rei[6]->SetAxis(m_PhantomScale * semiprincipalaxis);
  rei[6]->SetCenter(m_PhantomScale * (m_OriginOffset + center) );
  rei[6]->SetAngle(0.);
  rei[6]->SetDensity(0.01);
  rei[6]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.046;
  semiprincipalaxis[1] = 0.020;
  semiprincipalaxis[2] = 0.023;
  center[0] = 0.06;
  center[1] = -0.25;
  center[2] = -0.65;
  rei[7]->SetAxis(m_PhantomScale * semiprincipalaxis);
  rei[7]->SetCenter(m_PhantomScale * (m_OriginOffset + center) );
  rei[7]->SetAngle(90.);
  rei[7]->SetDensity(0.01);
  rei[7]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.056;
  semiprincipalaxis[1] = 0.010;
  semiprincipalaxis[2] = 0.040;
  center[0] = 0.060;
  center[1] = 0.625;
  center[2] = -0.105;
  rei[8]->SetAxis(m_PhantomScale * semiprincipalaxis);
  rei[8]->SetCenter(m_PhantomScale * (m_OriginOffset + center) );
  rei[8]->SetAngle(90.);
  rei[8]->SetDensity(0.02);
  rei[8]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.056;
  semiprincipalaxis[1] = 0.100;
  semiprincipalaxis[2] = 0.056;
  center[0] = 0.;
  center[1] = 0.625;
  center[2] = 0.100;
  rei[9]->SetAxis(m_PhantomScale * semiprincipalaxis);
  rei[9]->SetCenter(m_PhantomScale * (m_OriginOffset + center) );
  rei[9]->SetAngle(0.);
  rei[9]->SetDensity(-0.02);
  rei[9]->SetFigure("Ellipsoid");

  for ( unsigned int i = 0; i < NumberOfFig; i++ )
    {
    rei[i]->SetNumberOfThreads(this->GetNumberOfThreads());
    rei[i]->SetGeometry( this->GetGeometry() );

    if (i==0)
      rei[i]->SetInput( this->GetInput() );
    else
      rei[i]->SetInput( rei[i-1]->GetOutput() );
    }

  //Update
  rei[NumberOfFig - 1]->Update();
  this->GraftOutput( rei[NumberOfFig - 1]->GetOutput() );
}
} // end namespace rtk

#endif
