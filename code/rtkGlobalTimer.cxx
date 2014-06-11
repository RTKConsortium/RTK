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
GlobalTimer::Pointer GlobalTimer:: m_Instance = ITK_NULLPTR;

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
  rtk::WatcherForTimer watcher(o);
  m_Watchers.push_back(watcher);
}

unsigned int
GlobalTimer
::Start(const char *id)
{
  m_TimeProbesCollectorBase.Start(id);
  return m_GlobalTimerProbesCollector.Start(id);
}

void
GlobalTimer
::Stop(unsigned int pos, const char *id)
{
  m_TimeProbesCollectorBase.Stop(id);
  m_GlobalTimerProbesCollector.Stop(pos);
}

void
GlobalTimer
::Report(std::ostream & os) const
{
  if (m_Verbose) m_GlobalTimerProbesCollector.Report(os);
  else m_TimeProbesCollectorBase.Report(os);
}

void
GlobalTimer
::Clear(void)
{
  m_TimeProbesCollectorBase.Clear();
  m_GlobalTimerProbesCollector.Clear();
}



} // end namespace itk
