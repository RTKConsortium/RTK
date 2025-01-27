/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef rtkSubSelectFromListImageFilter_h
#define rtkSubSelectFromListImageFilter_h

#include "rtkSubSelectImageFilter.h"
#include "rtkConstantImageSource.h"

namespace rtk
{
/** \class SubSelectFromListImageFilter
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <typename ProjectionStackType>
class ITK_TEMPLATE_EXPORT SubSelectFromListImageFilter : public SubSelectImageFilter<ProjectionStackType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(SubSelectFromListImageFilter);

  /** Standard class type alias. */
  using Self = SubSelectFromListImageFilter;
  using Superclass = SubSelectImageFilter<ProjectionStackType>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(SubSelectFromListImageFilter);

  void
  SetSelectedProjections(std::vector<bool> sprojs);

protected:
  SubSelectFromListImageFilter();
  ~SubSelectFromListImageFilter() = default;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkSubSelectFromListImageFilter.hxx"
#endif

#endif
