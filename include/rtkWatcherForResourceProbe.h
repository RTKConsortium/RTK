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
#ifndef rtkWatcherForResourceProbe_h
#define rtkWatcherForResourceProbe_h

#include <itkCommand.h>
#include <itkProcessObject.h>

namespace rtk
{
/** \class WatcherForResourceProbe
 * \brief Very light watcher to monitor Start and End events
 * on all filters
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */

using namespace itk;

class WatcherForResourceProbe
{
public:
  /** Constructor. Takes a ProcessObject to monitor and an optional
   * comment string that is prepended to each event message. */
  WatcherForResourceProbe(itk::ProcessObject * o);

  /** Copy constructor */
  WatcherForResourceProbe(const WatcherForResourceProbe &);

  /** operator=  */
  WatcherForResourceProbe &
  operator=(const WatcherForResourceProbe &);

  /** Destructor. */
  virtual ~WatcherForResourceProbe();

  /** Method to get the name of the class being monitored by this
   *  WatcherForResourceProbe */
  const char *
  GetNameOfClass()
  {
    return (m_Process ? m_Process->GetNameOfClass() : "None");
  }

  /** Methods to access member data */
  /** Get a pointer to the process object being watched. */
  const ProcessObject *
  GetProcess() const
  {
    return m_Process;
  }

protected:
  /** Callback method to show the StartEvent */
  virtual void
  StartFilter();

  /** Callback method to show the EndEvent */
  virtual void
  EndFilter();

  /** Callback method to show the DeleteEvent */
  virtual void
  DeleteFilter();


private:
  itk::ProcessObject * m_Process;

  using CommandType = SimpleMemberCommand<WatcherForResourceProbe>;
  CommandType::Pointer m_StartFilterCommand;
  CommandType::Pointer m_EndFilterCommand;
  CommandType::Pointer m_DeleteFilterCommand;

  unsigned long m_StartTag;
  unsigned long m_EndTag;
  unsigned long m_DeleteTag;
};


} // end namespace rtk

#endif
