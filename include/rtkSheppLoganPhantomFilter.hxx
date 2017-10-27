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

#ifndef rtkSheppLoganPhantomFilter_hxx
#define rtkSheppLoganPhantomFilter_hxx


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
void SheppLoganPhantomFilter< TInputImage, TOutputImage >::SetEllipsoid(typename REIType::Pointer rei,
                                                                        double spax,
                                                                        double spay,
                                                                        double spaz,
                                                                        double centerx,
                                                                        double centery,
                                                                        double centerz,
                                                                        double angle,
                                                                        double density)
{
  VectorType semiprincipalaxis, center;
  semiprincipalaxis[0] = spax * m_PhantomScale[0];
  semiprincipalaxis[1] = spay * m_PhantomScale[1];
  semiprincipalaxis[2] = spaz * m_PhantomScale[2];
  center[0] = (centerx + m_OriginOffset[0]) * m_PhantomScale[0];
  center[1] = (centery + m_OriginOffset[1]) * m_PhantomScale[1];
  center[2] = (centerz + m_OriginOffset[2]) * m_PhantomScale[2];
  rei->SetAxis(semiprincipalaxis);
  rei->SetCenter(center);
  rei->SetAngle(angle);
  rei->SetDensity(density);
  rei->SetFigure("Ellipsoid");
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

  SetEllipsoid(rei[0], 0.69, 0.90, 0.92, 0., 0., 0., 0., 2.);
  SetEllipsoid(rei[1], 0.6624, 0.880, 0.874, 0., 0., 0., 0., -0.98);
  SetEllipsoid(rei[2], 0.41, 0.21, 0.16, -0.22, -0.25, 0., 108., -0.02);
  SetEllipsoid(rei[3], 0.31, 0.22, 0.11, 0.22, -0.25, 0., 72., -0.02);
  SetEllipsoid(rei[4], 0.21, 0.50, 0.25, 0., -0.25, 0.35, 0., 0.02);
  SetEllipsoid(rei[5], 0.046, 0.046, 0.046, 0., -0.25, 0.10, 0., 0.02);
  SetEllipsoid(rei[6], 0.046, 0.02, 0.023, -0.08, -0.25, -0.65, 0., 0.01);
  SetEllipsoid(rei[7], 0.046, 0.02, 0.023, 0.06, -0.25, -0.65, 90., 0.01);
  SetEllipsoid(rei[8], 0.056, 0.01, 0.04, 0.06, 0.625, -0.105, 90., 0.02);
  SetEllipsoid(rei[9], 0.056, 0.1, 0.056, 0., 0.625, 0.1, 0., -0.02);

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
