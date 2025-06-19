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

#ifndef rtkGeneralPurposeFunctions_h
#define rtkGeneralPurposeFunctions_h

#include <vector>


#include "math.h"

#include <itkMacro.h>
#include <itkImageFileWriter.h>
#include <itkMath.h>

namespace rtk
{

/** \brief A few functions that are used either in the applications or for debugging purposes
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */

static inline std::vector<double>
ReadSignalFile(std::string filename)
{
  std::vector<double> signalVector;
  std::ifstream       is(filename.c_str());
  if (!is.is_open())
  {
    itkGenericExceptionMacro(<< "Could not open signal file " << filename);
  }

  double      value = NAN;
  std::string s;
  while (getline(is, s))
  {
    if (!s.empty())
    {
      std::istringstream tmp(s);
      tmp >> value;
      if (itk::Math::Round<double>(value * 100) / 100 == 1)
        signalVector.push_back(0);
      else
        signalVector.push_back(itk::Math::Round<double>(value * 100) / 100);
    }
  }

  return signalVector;
}

template <typename ImageType>
void
WriteImage(typename ImageType::ConstPointer input, std::string name)
{
  // Create an itk::ImageFileWriter
  using WriterType = itk::ImageFileWriter<ImageType>;
  auto writer = WriterType::New();
  writer->SetInput(input);
  writer->SetFileName(name);
  writer->Update();
}

} // namespace rtk

#endif // rtkGeneralPurposeFunctions_h
