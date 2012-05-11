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

#ifndef __rtkConfigFileReader_h
#define __rtkConfigFileReader_h

#include <itkNumericTraits.h>
#include <vector>
#include <itkImageBase.h>
#include "rtkRayQuadricIntersectionFunction.h"
#include "rtkMacro.h"

namespace rtk
{

/** \class ConfigFileReader
 * \brief Reads configuration file which contains
 * the specifications for a specific phantom figure.
 * \ingroup Functions
 */
class ConfigFileReader :
    public itk::Object
{
public:
  /** Standard class typedefs. */
  typedef ConfigFileReader  Self;
  typedef itk::Object                              Superclass;
  typedef itk::SmartPointer<Self>                  Pointer;
  typedef itk::SmartPointer<const Self>            ConstPointer;
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConfigFileReader, itk::Object);

  /** Useful defines. */
  typedef std::vector<double> VectorType;
  typedef std::vector< std::vector<double> > VectorOfVectorType;

  bool Config( const std::string input);

  rtkSetMacro(Fig, VectorOfVectorType);
  rtkGetMacro(Fig, VectorOfVectorType);

protected:

  /// Constructor
  ConfigFileReader() {};

  /// Destructor
  ~ConfigFileReader() {};

  /** Corners of the image Quadric */
  VectorOfVectorType m_Fig;

private:
  ConfigFileReader( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConfigFileReader.txx"
#endif

#endif
