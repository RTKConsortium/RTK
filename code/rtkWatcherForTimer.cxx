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
#include "rtkWatcherForTimer.h"
#include "rtkGlobalTimer.h"

namespace rtk
{
WatcherForTimer
::WatcherForTimer(ProcessObject *o)
{
  // Initialize state
  m_Process = o;

  // Create a series of commands
  m_StartFilterCommand =      CommandType::New();
  m_EndFilterCommand =        CommandType::New();

  // Assign the callbacks
  m_StartFilterCommand->SetCallbackFunction(this,
                                            &WatcherForTimer::StartFilter);
  m_EndFilterCommand->SetCallbackFunction(this,
                                          &WatcherForTimer::EndFilter);

  // Add the commands as observers
  m_StartTag = m_Process->AddObserver(StartEvent(), m_StartFilterCommand);
  m_EndTag = m_Process->AddObserver(EndEvent(), m_EndFilterCommand);
}

WatcherForTimer
::WatcherForTimer() :
  m_Process(NULL),
  m_StartTag(0),
  m_EndTag(0)
{
}

void
WatcherForTimer
::StartFilter()
{
  m_IndexInGlobalTimer = rtk::GlobalTimer::GetInstance()->Start(m_Process->GetNameOfClass());
}

void
WatcherForTimer
::EndFilter()
{
  rtk::GlobalTimer::GetInstance()->Stop(m_IndexInGlobalTimer, m_Process->GetNameOfClass());
}

WatcherForTimer
::WatcherForTimer(const WatcherForTimer & watch)
{
  // Remove any observers we have on the old process object
  if ( m_Process )
    {
    if ( m_StartFilterCommand )
      {
      m_Process->RemoveObserver(m_StartTag);
      }
    if ( m_EndFilterCommand )
      {
      m_Process->RemoveObserver(m_EndTag);
      }
    }

  // Initialize state
  m_Process = watch.m_Process;
  m_StartTag = 0;
  m_EndTag = 0;

  // Create a series of commands
  if ( m_Process )
    {
    m_StartFilterCommand =      CommandType::New();
    m_EndFilterCommand =        CommandType::New();

    // Assign the callbacks
    m_StartFilterCommand->SetCallbackFunction(this,
                                              &WatcherForTimer::StartFilter);
    m_EndFilterCommand->SetCallbackFunction(this,
                                            &WatcherForTimer::EndFilter);

    // Add the commands as observers
    m_StartTag = m_Process->AddObserver(StartEvent(), m_StartFilterCommand);
    m_EndTag = m_Process->AddObserver(EndEvent(), m_EndFilterCommand);
    }
}

WatcherForTimer &
WatcherForTimer
::operator=(const WatcherForTimer & watch)
{
  if(this != &watch)
    {
    // Remove any observers we have on the old process object
    if ( m_Process )
      {
      if ( m_StartFilterCommand )
        {
        m_Process->RemoveObserver(m_StartTag);
        }
      if ( m_EndFilterCommand )
        {
        m_Process->RemoveObserver(m_EndTag);
        }
      }

    // Initialize state
    m_Process = watch.m_Process;

    m_StartTag = 0;
    m_EndTag = 0;

    // Create a series of commands
    if ( m_Process )
      {
      m_StartFilterCommand =      CommandType::New();
      m_EndFilterCommand =        CommandType::New();

      // Assign the callbacks
      m_StartFilterCommand->SetCallbackFunction(this,
                                                &WatcherForTimer::StartFilter);
      m_EndFilterCommand->SetCallbackFunction(this,
                                              &WatcherForTimer::EndFilter);

      // Add the commands as observers
      m_StartTag = m_Process->AddObserver(StartEvent(), m_StartFilterCommand);
      m_EndTag = m_Process->AddObserver(EndEvent(), m_EndFilterCommand);
      }
    }
  return *this;
}

WatcherForTimer
::~WatcherForTimer()
{
  // Remove any observers we have on the old process object
  if ( m_Process )
    {
    if ( m_StartFilterCommand )
      {
      m_Process->RemoveObserver(m_StartTag);
      }
    if ( m_EndFilterCommand )
      {
      m_Process->RemoveObserver(m_EndTag);
      }
    }
}
} // end namespace itk
