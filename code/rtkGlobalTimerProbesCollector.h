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
#ifndef rtkGlobalTimerProbesCollector_h
#define rtkGlobalTimerProbesCollector_h


#include "itkTimeProbe.h"
#include "itkMemoryUsageObserver.h"

namespace rtk
{
/** \class GlobalTimerProbesCollector
 *  \brief Aggregates a set of probes.
 *
 *  This class defines a set of ResourceProbes and assign names to them.
 *  The user can start and stop each one of the probes by addressing them by name.
 *
 *  \sa ResourceProbe
 *
 * \ingroup ITKCommon
 */

class GlobalTimerProbesCollector
{
public:
  typedef std::string                    IdType;
  typedef std::vector< IdType >          IdVector;
  typedef std::vector< itk::TimeProbe >  ProbeVector;
  typedef std::vector< unsigned int >    IndentVector;

  /** constructor */
  GlobalTimerProbesCollector();

  /** destructor */
  virtual ~GlobalTimerProbesCollector();

  /** Start a probe with a particular name. If the time probe does not
   * exist, it will be created */
  virtual unsigned int Start(const char *name);

  /** Stop a time probe identified with a name */
  virtual void Stop(unsigned int pos);

  /** Report the summary of results from the probes */
  virtual void Report(std::ostream & os = std::cout) const;

  /** Destroy the set of probes. New probes can be created after invoking this
    method. */
  virtual void Clear(void);

protected:
  ProbeVector  m_Probes;
  IdVector     m_Ids;
  IndentVector m_Indent;
  unsigned int m_CurrentIndent;
};
} // end namespace itk

#endif //rtkGlobalTimerProbesCollector_h
