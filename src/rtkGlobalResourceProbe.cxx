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
#include "rtkGlobalResourceProbe.h"
#include "itkObjectFactory.h"

namespace rtk
{
GlobalResourceProbe::Pointer GlobalResourceProbe::m_Instance = nullptr;

/**
 * Prompting off by default
 */
GlobalResourceProbe ::GlobalResourceProbe()
{
  m_Verbose = false;
}

GlobalResourceProbe ::~GlobalResourceProbe()
{
  if (m_Verbose)
    this->Report(std::cout);
  this->Clear();
}

void
GlobalResourceProbe ::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "GlobalResourceProbe (single instance): " << (void *)GlobalResourceProbe::m_Instance << std::endl;
}

/**
 * Return the single instance of the GlobalResourceProbe
 */
GlobalResourceProbe::Pointer
GlobalResourceProbe ::GetInstance()
{
  if (!GlobalResourceProbe::m_Instance)
  {
    // Try the factory first
    GlobalResourceProbe::m_Instance = ObjectFactory<Self>::Create();
    // if the factory did not provide one, then create it here
    if (!GlobalResourceProbe::m_Instance)
    {
      GlobalResourceProbe::m_Instance = new GlobalResourceProbe;
      // Remove extra reference from construction.
      GlobalResourceProbe::m_Instance->UnRegister();
    }
  }
  /**
   * return the instance
   */
  return GlobalResourceProbe::m_Instance;
}

/**
 * This just calls GetInstance
 */
GlobalResourceProbe::Pointer
GlobalResourceProbe ::New()
{
  return GetInstance();
}

void
GlobalResourceProbe ::Watch(ProcessObject * o)
{
  m_Mutex.lock();
  auto * w = new rtk::WatcherForResourceProbe(o);
  m_Watchers.push_back(w);
  m_Mutex.unlock();
}

void
GlobalResourceProbe ::Remove(const rtk::WatcherForResourceProbe * w)
{
  m_Mutex.lock();
  auto itw = std::find(m_Watchers.begin(), m_Watchers.end(), w);
  if (itw != m_Watchers.end())
  {
    delete *itw;
    m_Watchers.erase(itw);
  }
  m_Mutex.unlock();
}

void
GlobalResourceProbe ::Start(const char * id)
{
  m_Mutex.lock();
  m_ResourceProbesCollector.Start(id);
  m_Mutex.unlock();
}

void
GlobalResourceProbe ::Stop(const char * id)
{
  m_Mutex.lock();
  m_ResourceProbesCollector.Stop(id);
  m_Mutex.unlock();
}

void
GlobalResourceProbe ::Report(std::ostream & os) const
{
  m_ResourceProbesCollector.Report(os);
}

void
GlobalResourceProbe ::Clear()
{
  m_Mutex.lock();
  m_ResourceProbesCollector.Clear();
  m_Watchers.clear();
  m_Mutex.unlock();
}
} // end namespace rtk
