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
#ifndef rtkTimeProbesCollectorBase_h
#define rtkTimeProbesCollectorBase_h

#include <itkTimeProbesCollectorBase.h>

namespace rtk
{
/** \class TimeProbesCollectorBase
 *  \brief Aggregates a set of time probes.
 *
 *  Derives from itk::TimeProbesCollectorBase but improves the report output.
 */
class ITKCommon_EXPORT TimeProbesCollectorBase: public itk::TimeProbesCollectorBase
{
public:
  /** Report the summary of results from the probes */
  virtual void ConstReport(std::ostream & os = std::cout) const;
};
}

#endif //rtkTimeProbesCollectorBase_h
