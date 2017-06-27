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
#ifndef rtkSelectOneProjectionPerCycleImageFilter_h
#define rtkSelectOneProjectionPerCycleImageFilter_h

#include "rtkSubSelectImageFilter.h"

namespace rtk
{
/** \class SelectOneProjectionPerCycleImageFilter
 * \brief Subselects one projection per respiratory cycle from a phase signal
 *
 * The selection selects for each respiratory cycle the projection that is the
 * nearest to the phase parameter. The phase signal is assumed to be between
 * 0 and 1 and monotonic.
 *
 * \test rtkselectoneprojpercycletest.cxx
 *
 * \author Simon Rit
 */
template< typename ProjectionStackType>
class SelectOneProjectionPerCycleImageFilter : public SubSelectImageFilter<ProjectionStackType>
{
public:
  /** Standard class typedefs. */
  typedef SelectOneProjectionPerCycleImageFilter    Self;
  typedef SubSelectImageFilter<ProjectionStackType> Superclass;
  typedef itk::SmartPointer< Self >                 Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(SelectOneProjectionPerCycleImageFilter, SubSelectImageFilter)

  /** File name of a text file with one phase value between 0 and 1 per line. */
  itkGetMacro(SignalFilename, std::string);
  itkSetMacro(SignalFilename, std::string);

  /** Phase value for which we want the closest projection image per cycle. */
  itkSetMacro(Phase, double)
  itkGetMacro(Phase, double)

protected:
  SelectOneProjectionPerCycleImageFilter();
  ~SelectOneProjectionPerCycleImageFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;

private:
  SelectOneProjectionPerCycleImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);                         //purposely not implemented

  std::string         m_SignalFilename;
  double              m_Phase;
  std::vector<double> m_Signal;
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSelectOneProjectionPerCycleImageFilter.hxx"
#endif

#endif
