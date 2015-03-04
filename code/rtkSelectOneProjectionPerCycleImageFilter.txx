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
#ifndef __rtkSelectOneProjectionPerCycleImageFilter_txx
#define __rtkSelectOneProjectionPerCycleImageFilter_txx

#include "rtkSelectOneProjectionPerCycleImageFilter.h"

namespace rtk
{

template<typename ProjectionStackType>
SelectOneProjectionPerCycleImageFilter<ProjectionStackType>
::SelectOneProjectionPerCycleImageFilter():
  m_Phase(0.)
{
}

template<typename ProjectionStackType>
void
SelectOneProjectionPerCycleImageFilter<ProjectionStackType>
::SetSignalFilename (const std::string _arg)
{
  itkDebugMacro("setting SignalFilename to " << _arg);
  if ( this->m_SignalFilename != _arg )
    {
    this->m_SignalFilename = _arg;
    this->Modified();

    std::ifstream is( _arg.c_str() );
    if( !is.is_open() )
      {
      itkGenericExceptionMacro(<< "Could not open signal file " << m_SignalFilename);
      }

    double value;
    while( !is.eof() )
      {
      is >> value;
      m_Signal.push_back(value);
      }
    }
}

template<typename ProjectionStackType>
void SelectOneProjectionPerCycleImageFilter<ProjectionStackType>
::GenerateOutputInformation()
{
  this->m_NbSelectedProjs = 0;
  this->m_SelectedProjections.resize(m_Signal.size());
  std::fill(this->m_SelectedProjections.begin(), this->m_SelectedProjections.end(), false);
  for(unsigned int i=0; i<m_Signal.size()-1; i++)
    {
    // Put phase between -0.5 and 0.5 with 0 the ref phase
    double valPrev = m_Signal[i]-m_Phase;
    valPrev -= vnl_math_floor(valPrev);
    if(valPrev>0.5)
      valPrev -= 1.;
    double valAfter = m_Signal[i+1]-m_Phase;
    valAfter -= vnl_math_floor(valAfter);
    if(valAfter>0.5)
      valAfter -= 1.;

    // Frame is selected if phase is increasing and has opposite signs
    if(valPrev<valAfter && valAfter * valPrev <= 0.)
      {
      if(vnl_math_abs(valPrev)>vnl_math_abs(valAfter))
        {
        this->m_SelectedProjections[i+1] = true;
        this->m_NbSelectedProjs++;
        }
      else if(!this->m_SelectedProjections[i])
        {
        this->m_SelectedProjections[i] = true;
        this->m_NbSelectedProjs++;
        }
      }
    }

  Superclass::GenerateOutputInformation();
}

}// end namespace


#endif
