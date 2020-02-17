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
#include "rtkWatcherForResourceProbe.h"
#include "rtkGlobalResourceProbe.h"

namespace rtk
{
WatcherForResourceProbe ::WatcherForResourceProbe(ProcessObject * o)
{
  // Initialize state
  m_Process = o;

  // Create a series of commands
  m_StartFilterCommand = CommandType::New();
  m_EndFilterCommand = CommandType::New();
  m_DeleteFilterCommand = CommandType::New();

  // Assign the callbacks
  m_StartFilterCommand->SetCallbackFunction(this, &WatcherForResourceProbe::StartFilter);
  m_EndFilterCommand->SetCallbackFunction(this, &WatcherForResourceProbe::EndFilter);
  m_DeleteFilterCommand->SetCallbackFunction(this, &WatcherForResourceProbe::DeleteFilter);

  // Add the commands as observers
  m_StartTag = m_Process->AddObserver(StartEvent(), m_StartFilterCommand);
  m_EndTag = m_Process->AddObserver(EndEvent(), m_EndFilterCommand);
  m_DeleteTag = m_Process->AddObserver(DeleteEvent(), m_DeleteFilterCommand);
}

void
WatcherForResourceProbe ::StartFilter()
{
  rtk::GlobalResourceProbe::GetInstance()->Start(m_Process->GetNameOfClass());
}

void
WatcherForResourceProbe ::EndFilter()
{
  rtk::GlobalResourceProbe::GetInstance()->Stop(m_Process->GetNameOfClass());
}

void
WatcherForResourceProbe ::DeleteFilter()
{
  if (m_StartFilterCommand)
  {
    m_Process->RemoveObserver(m_StartTag);
  }
  if (m_EndFilterCommand)
  {
    m_Process->RemoveObserver(m_EndTag);
  }
  if (m_DeleteFilterCommand)
  {
    m_Process->RemoveObserver(m_DeleteTag);
  }
  rtk::GlobalResourceProbe::GetInstance()->Remove(this);
}

WatcherForResourceProbe ::WatcherForResourceProbe(const WatcherForResourceProbe & watch)
{
  // Remove any observers we have on the old process object
  if (m_Process)
  {
    if (m_StartFilterCommand)
    {
      m_Process->RemoveObserver(m_StartTag);
    }
    if (m_EndFilterCommand)
    {
      m_Process->RemoveObserver(m_EndTag);
    }
    if (m_DeleteFilterCommand)
    {
      m_Process->RemoveObserver(m_DeleteTag);
    }
  }

  // Initialize state
  m_Process = watch.m_Process;
  m_StartTag = 0;
  m_EndTag = 0;
  m_DeleteTag = 0;

  // Create a series of commands
  if (m_Process)
  {
    m_StartFilterCommand = CommandType::New();
    m_EndFilterCommand = CommandType::New();
    m_DeleteFilterCommand = CommandType::New();

    // Assign the callbacks
    m_StartFilterCommand->SetCallbackFunction(this, &WatcherForResourceProbe::StartFilter);
    m_EndFilterCommand->SetCallbackFunction(this, &WatcherForResourceProbe::EndFilter);
    m_DeleteFilterCommand->SetCallbackFunction(this, &WatcherForResourceProbe::DeleteFilter);

    // Add the commands as observers
    m_StartTag = m_Process->AddObserver(StartEvent(), m_StartFilterCommand);
    m_EndTag = m_Process->AddObserver(EndEvent(), m_EndFilterCommand);
    m_DeleteTag = m_Process->AddObserver(DeleteEvent(), m_DeleteFilterCommand);
  }
}

WatcherForResourceProbe &
WatcherForResourceProbe ::operator=(const WatcherForResourceProbe & watch)
{
  if (this != &watch)
  {
    // Remove any observers we have on the old process object
    if (m_Process)
    {
      if (m_StartFilterCommand)
      {
        m_Process->RemoveObserver(m_StartTag);
      }
      if (m_EndFilterCommand)
      {
        m_Process->RemoveObserver(m_EndTag);
      }
      if (m_DeleteFilterCommand)
      {
        m_Process->RemoveObserver(m_DeleteTag);
      }
    }

    // Initialize state
    m_Process = watch.m_Process;

    m_StartTag = 0;
    m_EndTag = 0;
    m_DeleteTag = 0;

    // Create a series of commands
    if (m_Process)
    {
      m_StartFilterCommand = CommandType::New();
      m_EndFilterCommand = CommandType::New();
      m_DeleteFilterCommand = CommandType::New();

      // Assign the callbacks
      m_StartFilterCommand->SetCallbackFunction(this, &WatcherForResourceProbe::StartFilter);
      m_EndFilterCommand->SetCallbackFunction(this, &WatcherForResourceProbe::EndFilter);
      m_DeleteFilterCommand->SetCallbackFunction(this, &WatcherForResourceProbe::DeleteFilter);

      // Add the commands as observers
      m_StartTag = m_Process->AddObserver(StartEvent(), m_StartFilterCommand);
      m_EndTag = m_Process->AddObserver(EndEvent(), m_EndFilterCommand);
      m_DeleteTag = m_Process->AddObserver(DeleteEvent(), m_DeleteFilterCommand);
    }
  }
  return *this;
}

WatcherForResourceProbe ::~WatcherForResourceProbe() = default;

} // namespace rtk
