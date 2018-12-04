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

#ifndef rtkThreeDCircularProjectionGeometryXMLFileWriter_h
#define rtkThreeDCircularProjectionGeometryXMLFileWriter_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "RTKExport.h"
#include <itkXMLFile.h>
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

/** \class ThreeDCircularProjectionGeometryXMLFileWriter
 *
 * Writes an XML-format file containing geometry for reconstruction
 *
 * \author Simon Rit
 *
 * \ingroup RTK IOFilters
 */
class RTK_EXPORT ThreeDCircularProjectionGeometryXMLFileWriter :
  public itk::XMLWriterBase< ThreeDCircularProjectionGeometry >
{
public:
  /** standard typedefs */
  typedef itk::XMLWriterBase< ThreeDCircularProjectionGeometry > Superclass;
  typedef ThreeDCircularProjectionGeometryXMLFileWriter          Self;
  typedef itk::SmartPointer<Self>                                Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ThreeDCircularProjectionGeometryXMLFileWriter, itk::XMLWriterBase)

  /** Test whether a file is writable. */
  int CanWriteFile(const char* name) ITK_OVERRIDE;

  /** Actually write out the file in question */
  int WriteFile() ITK_OVERRIDE;

protected:
  ThreeDCircularProjectionGeometryXMLFileWriter() {}
  ~ThreeDCircularProjectionGeometryXMLFileWriter() {}
  
  /** If all values are equal in v, write first value (if not 0.) in
      output file with parameter value s and return true. Return false
      otherwise.
   */
  bool WriteGlobalParameter(std::ofstream &output, const std::string &indent,
                            const std::vector<double> &v, const std::string &s,
                            bool convertToDegrees=false,
                            double defval=0.);

  /** Write projection specific parameter with name s. */
  void WriteLocalParameter(std::ofstream &output, const std::string &indent,
                           const double &v, const std::string &s);

private:
   //purposely not implemented
  ThreeDCircularProjectionGeometryXMLFileWriter(const Self&);
  void operator=(const Self&);

};
}

#endif
