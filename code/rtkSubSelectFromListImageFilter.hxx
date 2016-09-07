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
#ifndef rtkSubSelectFromListImageFilter_hxx
#define rtkSubSelectFromListImageFilter_hxx

#include "rtkSubSelectFromListImageFilter.h"

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>

#include "math.h"

namespace rtk
{

template<typename ProjectionStackType>
SubSelectFromListImageFilter<ProjectionStackType>::SubSelectFromListImageFilter()
{
}

template<typename ProjectionStackType>
void SubSelectFromListImageFilter<ProjectionStackType>::SetSelectedProjections(std::vector<bool> sprojs)
{
  // Set the selected projection boolean vector
  this->m_SelectedProjections = sprojs;

  // Update the number of selected projections
  this->m_NbSelectedProjs = 0;
  for (unsigned int i = 0; i < this->m_SelectedProjections.size(); i++)
      if (this->m_SelectedProjections[i]) this->m_NbSelectedProjs += 1;

  // Notify the filter that it has been modified
  this->Modified();
}

}// end namespace


#endif
