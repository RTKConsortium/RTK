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
#include "rtkGlobalTimer.h"
#include "itkObjectFactory.h"

namespace rtk
{
GlobalTimer::Pointer GlobalTimer::m_Instance = ITK_NULLPTR;

/**
 * Prompting off by default
 */
GlobalTimer
::GlobalTimer()
{
  m_Verbose=false;
}

GlobalTimer
::~GlobalTimer()
{
  if(m_Verbose)
    this->Report(std::cout);
  this->Clear();
}

void
GlobalTimer
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "GlobalTimer (single instance): "
     << (void *)GlobalTimer::m_Instance << std::endl;
}

/**
 * Return the single instance of the GlobalTimer
 */
GlobalTimer::Pointer
GlobalTimer
::GetInstance()
{
  if ( !GlobalTimer::m_Instance )
    {
    // Try the factory first
    GlobalTimer::m_Instance  = ObjectFactory< Self >::Create();
    // if the factory did not provide one, then create it here
    if ( !GlobalTimer::m_Instance )
      {
      GlobalTimer::m_Instance = new GlobalTimer;
      // Remove extra reference from construction.
      GlobalTimer::m_Instance->UnRegister();
      }
    }
  /**
   * return the instance
   */
  return GlobalTimer::m_Instance;
}

/**
 * This just calls GetInstance
 */
GlobalTimer::Pointer
GlobalTimer
::New()
{
  return GetInstance();
}

void
GlobalTimer
::Watch(ProcessObject *o)
{
  m_Mutex.Lock();
  rtk::WatcherForTimer *w = new rtk::WatcherForTimer(o);
  m_Watchers.push_back(w);
  m_Mutex.Unlock();
}

void
GlobalTimer
::Remove(const rtk::WatcherForTimer *w)
{
  m_Mutex.Lock();
  std::vector<rtk::WatcherForTimer*>::iterator itw = std::find( m_Watchers.begin(), m_Watchers.end(), w);
  if(itw != m_Watchers.end())
    {
    delete *itw;
    m_Watchers.erase(itw);
    }
  m_Mutex.Unlock();
}

void
GlobalTimer
::Start(const char *id)
{
  m_Mutex.Lock();
  m_TimeProbesCollectorBase.Start(id);
  m_Mutex.Unlock();
}

void
GlobalTimer
::Stop(const char *id)
{
  m_Mutex.Lock();
  m_TimeProbesCollectorBase.Stop(id);
  m_Mutex.Unlock();
}

void
GlobalTimer
::Report(std::ostream & os) const
{
  m_Mutex.Lock();
  m_TimeProbesCollectorBase.ConstReport(os);
  m_Mutex.Unlock();
}

void
GlobalTimer
::Clear(void)
{
  m_Mutex.Lock();
  m_TimeProbesCollectorBase.Clear();
  m_Watchers.clear();
  m_Mutex.Unlock();
}
} // end namespace rtk
