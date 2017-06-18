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

#include "rtkTimeProbesCollectorBase.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

void
rtk::TimeProbesCollectorBase
::ConstReport(std::ostream & os) const
{
  MapType::const_iterator probe = this->m_Probes.begin();
  MapType::const_iterator end   = this->m_Probes.end();

  if ( probe == end )
    {
    return;
    }

  unsigned int maxlength = sizeof("Probe Tag")/sizeof(char);
  while(probe != end)
    maxlength = std::max(maxlength, (unsigned int) (probe++)->first.size());
  maxlength += 2;
  probe = this->m_Probes.begin();

  os << std::endl << std::endl;
  os.width(maxlength+10+10+15);
  os << std::setfill('*') << "" << std::endl << std::setfill(' ');
  os.width(maxlength);
  os << std::left;
  os <<  "Probe Tag";
  os.width(10);
  os <<  "Starts";
  os.width(10);
  os <<  "Stops";
  os.width(15);
  os << std::string(probe->second.GetType() + " (" + probe->second.GetUnit() + ")");
  os << std::endl;
  os.width(maxlength+10+10+15);
  os << std::setfill('*') << "" << std::endl << std::setfill(' ');

  while ( probe != end )
    {
    os.width(maxlength);
    os <<  probe->first;
    os.width(10);
    os <<  probe->second.GetNumberOfStarts();
    os.width(10);
    os <<  probe->second.GetNumberOfStops();
    os.width(15);
    os <<  probe->second.GetMean();
    os << std::endl;
    probe++;
    }
  os << std::setfill('*') << "" << std::endl << std::setfill(' ');
}
