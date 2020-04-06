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
#ifndef rtkSelectOneProjectionPerCycleImageFilter_hxx
#define rtkSelectOneProjectionPerCycleImageFilter_hxx

#include "rtkSelectOneProjectionPerCycleImageFilter.h"
#include <itkCSVArray2DFileReader.h>

namespace rtk
{

template <typename ProjectionStackType>
SelectOneProjectionPerCycleImageFilter<ProjectionStackType>::SelectOneProjectionPerCycleImageFilter() = default;

template <typename ProjectionStackType>
void
SelectOneProjectionPerCycleImageFilter<ProjectionStackType>::GenerateOutputInformation()
{
  // Read signal file
  using ReaderType = itk::CSVArray2DFileReader<double>;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(m_SignalFilename);
  reader->SetFieldDelimiterCharacter(';');
  reader->HasRowHeadersOff();
  reader->HasColumnHeadersOff();
  reader->Update();
  m_Signal = reader->GetArray2DDataObject()->GetColumn(0);

  this->m_NbSelectedProjs = 0;
  this->m_SelectedProjections.resize(m_Signal.size());
  std::fill(this->m_SelectedProjections.begin(), this->m_SelectedProjections.end(), false);
  for (unsigned int i = 0; i < m_Signal.size() - 1; i++)
  {
    // Put phase between -0.5 and 0.5 with 0 the ref phase
    double valPrev = m_Signal[i] - m_Phase;
    valPrev -= itk::Math::floor(valPrev);
    if (valPrev > 0.5)
      valPrev -= 1.;
    double valAfter = m_Signal[i + 1] - m_Phase;
    valAfter -= itk::Math::floor(valAfter);
    if (valAfter > 0.5)
      valAfter -= 1.;

    // Frame is selected if phase is increasing and has opposite signs
    if (valPrev < valAfter && valAfter * valPrev <= 0.)
    {
      if (itk::Math::abs(valPrev) > itk::Math::abs(valAfter))
      {
        this->m_SelectedProjections[i + 1] = true;
        this->m_NbSelectedProjs++;
      }
      else if (!this->m_SelectedProjections[i])
      {
        this->m_SelectedProjections[i] = true;
        this->m_NbSelectedProjs++;
      }
    }
  }

  Superclass::GenerateOutputInformation();
}

} // namespace rtk


#endif
