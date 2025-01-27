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

#ifndef rtkThreeDCircularProjectionGeometryXMLFileWriter_h
#define rtkThreeDCircularProjectionGeometryXMLFileWriter_h

#ifdef _MSC_VER
#  pragma warning(disable : 4786)
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
class RTK_EXPORT ThreeDCircularProjectionGeometryXMLFileWriter
  : public itk::XMLWriterBase<ThreeDCircularProjectionGeometry>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ThreeDCircularProjectionGeometryXMLFileWriter);

  /** standard type alias */
  using Superclass = itk::XMLWriterBase<ThreeDCircularProjectionGeometry>;
  using Self = ThreeDCircularProjectionGeometryXMLFileWriter;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ThreeDCircularProjectionGeometryXMLFileWriter);

  /** Test whether a file is writable. */
  int
  CanWriteFile(const char * name) override;

  /** Actually write out the file in question */
  int
  WriteFile() override;

protected:
  ThreeDCircularProjectionGeometryXMLFileWriter() = default;
  ~ThreeDCircularProjectionGeometryXMLFileWriter() override = default;

  /** If all values are equal in v, write first value (if not 0.) in
      output file with parameter value s and return true. Return false
      otherwise.
   */
  bool
  WriteGlobalParameter(std::ofstream &             output,
                       const std::string &         indent,
                       const std::vector<double> & v,
                       const std::string &         s,
                       bool                        convertToDegrees = false,
                       double                      defval = 0.);

  /** Write projection specific parameter with name s. */
  void
  WriteLocalParameter(std::ofstream & output, const std::string & indent, const double & v, const std::string & s);
};

/** Convenience function for writing an geometry.
 *
 * The geometry parameter may be a either SmartPointer or a raw pointer and const or non-const.
 * */
template <typename TGeometryPointer>
ITK_TEMPLATE_EXPORT void
WriteGeometry(TGeometryPointer && geometry, const std::string & filename)
{
  using NonReferenceImagePointer = std::remove_reference_t<TGeometryPointer>;
  static_assert(std::is_pointer<NonReferenceImagePointer>::value ||
                  itk::mpl::IsSmartPointer<NonReferenceImagePointer>::Value,
                "WriteGeometry requires a raw pointer or SmartPointer.");

  auto writer = ThreeDCircularProjectionGeometryXMLFileWriter::New();
  writer->SetObject(geometry);
  writer->SetFilename(filename);
  writer->WriteFile();
}

} // namespace rtk

#endif
