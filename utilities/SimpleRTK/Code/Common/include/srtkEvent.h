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
#ifndef srtkEvent_h
#define srtkEvent_h

#include "srtkCommon.h"
#include <ostream>

namespace rtk {
namespace simple {

/** \brief Events which can be observed from ProcessObject.
 *
 * For more information see the page \ref CommandPage.
 */
enum EventEnum {
  /** \brief Occurs for all event types.
   */
  srtkAnyEvent = 0,
  /** \brief Occurs after the process has been aborted, but before exiting the Execute method.
   */
  srtkAbortEvent = 1,
  /** \brief Occurs when the underlying itk::ProcessObject is deleted.
   */
  srtkDeleteEvent = 2,
  /** \brief Occurs at then end of normal processing.
   */
  srtkEndEvent = 3,
  /** \brief Occurs with some algorithms that run for a fixed or undetermined number of iterations.
   */
  srtkIterationEvent = 4,
  /** \brief Occurs when the progress changes in most process objects.
   */
  srtkProgressEvent = 5,
  /** \brief Occurs when then itk::ProcessObject is starting.
   */
  srtkStartEvent = 6,
  /** \brief Occurs when some filters change processing to a different
   * scale
   *
   * \note This event is a sub-event of the more general
   * IterationEvent. The general iteration event will also be invoked.
   */
  srtkMultiResolutionIterationEvent = 9,
  /** \brief Other events may fall into this enumeration.
   */
  srtkUserEvent = 7
};

#ifndef SWIG
SRTKCommon_EXPORT std::ostream& operator<<(std::ostream& os, const EventEnum k);
#endif

} // end namespace simple
} // end namespace itk

#endif
