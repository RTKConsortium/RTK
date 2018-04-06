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
#ifndef __rtkGlobalTimerProbesCollector_hxx
#define __rtkGlobalTimerProbesCollector_hxx

#include "rtkGlobalTimerProbesCollector.h"
#include <iostream>
#include <iomanip>

namespace rtk
{

GlobalTimerProbesCollector
::GlobalTimerProbesCollector()
{
  m_CurrentIndent = 0;
}

GlobalTimerProbesCollector
::~GlobalTimerProbesCollector()
{}

unsigned int GlobalTimerProbesCollector::Start(const char *id)
{
  // Create a new time probe, store it in m_Probes and start it
  itk::TimeProbe tp;
  m_Probes.push_back(tp);

  // Store the filter's name in m_Ids
  m_Ids.push_back(id);

  // Store the current indent and increase it for the next call
  m_Indent.push_back(m_CurrentIndent);
  m_CurrentIndent++;

  // Return the position of this probe in m_Probes
  int pos = m_Probes.size() - 1;
  m_Probes[pos].Start();
  return pos;
}

void
GlobalTimerProbesCollector
::Stop(unsigned int pos)
{
  if ( pos >= m_Probes.size() )
    {
    itkGenericExceptionMacro(<< "The probe index \"" << pos << "\" does not exist. It can not be stopped.");
    return;
    }
  m_Probes[pos].Stop();

  // Decrease the current indent
  m_CurrentIndent--;
}

void
GlobalTimerProbesCollector
::Report(std::ostream & os) const
{
  unsigned int pos=0;
  if ( m_Probes.size() == 0 )
    {
    os << "No probes have been created" << std::endl;
    return;
    }

  os << std::setiosflags(std::ios::left);
  os << std::setw(60) << "Probe Tag";
  os << std::setw(5) << m_Probes[0].GetType() << " (" << m_Probes[0].GetUnit() << ")";
  os << std::endl;

  while ( pos < m_Probes.size() )
    {
      for(unsigned int indent=0; indent < m_Indent[pos]; indent++)
        {
          os << "| ";
        }

    os << std::setw(3) << "-->" << std::setw(60 - 2*m_Indent[pos] - 3) << m_Ids[pos];
    os << std::setw(12) << m_Probes[pos].GetTotal();
    os << std::endl;
    pos++;
    }
}

void
GlobalTimerProbesCollector
::Clear(void)
{
  this->m_Probes.clear();
  this->m_Ids.clear();
  this->m_Indent.clear();
  this->m_CurrentIndent = 0;
}

} // end namespace itk

#endif //__rtkGlobalTimerProbesCollector_hxx
