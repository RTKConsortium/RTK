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
#ifndef rtkSubSelectFromListImageFilter_h
#define rtkSubSelectFromListImageFilter_h

#include "rtkSubSelectImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkPhaseReader.h"

namespace rtk
{
template< typename ProjectionStackType>
class SubSelectFromListImageFilter : public SubSelectImageFilter<ProjectionStackType>
{
public:
    /** Standard class typedefs. */
    typedef SubSelectFromListImageFilter              Self;
    typedef SubSelectImageFilter<ProjectionStackType> Superclass;
    typedef itk::SmartPointer< Self >                 Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(SubSelectFromListImageFilter, SubSelectImageFilter)

    void SetSelectedProjections(std::vector<bool> sprojs);

protected:
    SubSelectFromListImageFilter();
    ~SubSelectFromListImageFilter(){}

private:
    SubSelectFromListImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSubSelectFromListImageFilter.hxx"
#endif

#endif
