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
#ifndef rtkPhaseReader_hxx
#define rtkPhaseReader_hxx

#include "rtkPhaseReader.h"

#include "itksys/SystemTools.hxx"
#include <vcl_limits.h>

namespace rtk
{

PhaseReader::PhaseReader()
{
}

void PhaseReader::PrintSelf(std::ostream & os, itk::Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  for (unsigned int proj=0; proj<m_Phases.size(); proj++)
      os << this->m_Phases[proj] << std::endl;
}

void PhaseReader::Parse()
{
  this->m_InputStream.clear();
  this->m_InputStream.open(this->m_FileName.c_str());
  if ( this->m_InputStream.fail() )
    {
    itkExceptionMacro(
                "The file " << this->m_FileName <<" cannot be opened for reading!"
                << std::endl
                << "Reason: "
                << itksys::SystemTools::GetLastSystemError() );
    }

  // Prepare to parse the file
  itk::SizeValueType rows, columns;
  this->GetDataDimension(rows,columns);
  if ( columns > 1 )
    {
    itkExceptionMacro(
                "The file " << this->m_FileName <<" should have only one column"
                << std::endl);
    }
  unsigned NumberOfProjections = rows;

  // parse the numeric data
  m_Phases.clear();
  std::string entry;
  for (unsigned int j = 0; j < NumberOfProjections; j++)
    {
    this->GetNextField(entry);
    m_Phases.push_back(atof(entry.c_str()));
    }

  this->m_InputStream.close();
}

/** Update method */

void PhaseReader::Update()
{
  this->Parse();
}

/** Get the output */

std::vector<float> PhaseReader::GetOutput()
{
  return this->m_Phases;
}

} //end namespace rtk

#endif
