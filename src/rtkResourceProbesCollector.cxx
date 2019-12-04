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

#include "rtkResourceProbesCollector.h"
#include <iostream>

namespace rtk
{

void
ResourceProbesCollector::Start(const char * id)
{
  // if the probe does not exist yet, it is created.
  this->m_TimeProbes[id].SetNameOfProbe(id);
  this->m_TimeProbes[id].Start();
  this->m_MemoryProbes[id].SetNameOfProbe(id);
  this->m_MemoryProbes[id].Start();
#ifdef RTK_USE_CUDA
  this->m_CudaMemoryProbes[id].SetNameOfProbe(id);
  this->m_CudaMemoryProbes[id].Start();
#endif
}


void
ResourceProbesCollector::Stop(const char * id)
{
  IdType tid = id;

  auto pos = this->m_TimeProbes.find(tid);
  if (pos == this->m_TimeProbes.end())
  {
    itkGenericExceptionMacro(<< "The probe \"" << id << "\" does not exist. It can not be stopped.");
    return;
  }
  pos->second.Stop();
  m_MemoryProbes[tid].Stop();
#ifdef RTK_USE_CUDA
  m_CudaMemoryProbes[tid].Stop();
#endif
}


void
ResourceProbesCollector::Report(std::ostream & os) const
{
  auto tProbe = this->m_TimeProbes.begin();
  auto tEnd = this->m_TimeProbes.end();

  if (tProbe == tEnd)
  {
    return;
  }

  unsigned int maxlength = sizeof("Probe Tag") / sizeof(char);
  while (tProbe != tEnd)
    maxlength = std::max(maxlength, (unsigned int)(tProbe++)->first.size());
  maxlength += 2;
  tProbe = this->m_TimeProbes.begin();

  auto mProbe = this->m_MemoryProbes.begin();
#ifdef RTK_USE_CUDA
  auto cProbe = this->m_CudaMemoryProbes.begin();
#endif
  os << std::endl << std::endl;
#ifdef RTK_USE_CUDA
  os.width(maxlength + 10 + 10 + 15 + 15 + 15);
#else
  os.width(maxlength + 10 + 10 + 15 + 15);
#endif
  os << std::setfill('*') << "" << std::endl << std::setfill(' ');
  os.width(maxlength);
  os << std::left;
  os << "Probe Tag";
  os.width(10);
  os << "Starts";
  os.width(10);
  os << "Stops";
  os.width(15);
  os << std::string(tProbe->second.GetType() + " (" + tProbe->second.GetUnit() + ")");
  os.width(15);
  os << std::string(mProbe->second.GetType() + " (" + mProbe->second.GetUnit() + ")");
#ifdef RTK_USE_CUDA
  os.width(15);
  os << std::string(cProbe->second.GetType() + " (" + cProbe->second.GetUnit() + ")");
#endif
  os << std::endl;
#ifdef RTK_USE_CUDA
  os.width(maxlength + 10 + 10 + 15 + 15 + 15);
#else
  os.width(maxlength + 10 + 10 + 15 + 15);
#endif
  os << std::setfill('*') << "" << std::endl << std::setfill(' ');

  while (tProbe != tEnd)
  {
    os.width(maxlength);
    os << tProbe->first;
    os.width(10);
    os << tProbe->second.GetNumberOfStarts();
    os.width(10);
    os << tProbe->second.GetNumberOfStops();
    os.width(15);
    os << tProbe++->second.GetMean();
    os.width(15);
    os << mProbe++->second.GetMean();
#ifdef RTK_USE_CUDA
    os.width(15);
    os << cProbe++->second.GetMean();
#endif
    os << std::endl;
  }
  os << std::setfill('*') << "" << std::endl << std::setfill(' ');
}

void
ResourceProbesCollector::Clear()
{
  this->m_TimeProbes.clear();
  this->m_MemoryProbes.clear();
#ifdef RTK_USE_CUDA
  this->m_CudaMemoryProbes.clear();
#endif
}


} // end namespace rtk
