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

#ifndef rtkResourceProbesCollector_h
#define rtkResourceProbesCollector_h

#include "rtkConfiguration.h"
#include <itkTimeProbe.h>
#include <itkMemoryProbe.h>

#ifdef RTK_USE_CUDA
#  include <itkCudaMemoryProbe.h>
#endif

namespace rtk
{
/** \class ResourceProbesCollector
 *  \brief Aggregates a set of time, memory and cuda memory probes.
 *
 * \ingroup RTK
 */
class ResourceProbesCollector
{
public:
  using IdType = std::string;
  using TimeMapType = std::map<IdType, itk::TimeProbe>;
  using MemoryMapType = std::map<IdType, itk::MemoryProbe>;
#ifdef RTK_USE_CUDA
  using CudaMemoryMapType = std::map<IdType, itk::CudaMemoryProbe>;
#endif

  /** destructor */
  virtual ~ResourceProbesCollector() = default;

  /** Start a probe with a particular name. If the time probe does not
   * exist, it will be created */
  virtual void
  Start(const char * id);

  /** Stop a time probe identified with a name */
  virtual void
  Stop(const char * id);

  /** Report the summary of results from all probes */
  virtual void
  Report(std::ostream & os = std::cout) const;

  /** Destroy the set of probes. New probes can be created after invoking this
    method. */
  virtual void
  Clear();

protected:
  TimeMapType   m_TimeProbes;
  MemoryMapType m_MemoryProbes;
#ifdef RTK_USE_CUDA
  CudaMemoryMapType m_CudaMemoryProbes;
#endif
};
} // namespace rtk

#endif // rtkResourceProbesCollector_h
