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

#include "rtkorageometry_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkOraGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

int main(int argc, char * argv[])
{
  GGO(rtkorageometry, args_info);

  // Generate file names of projections
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(args_info.nsort_flag);
  names->SetRegularExpression(args_info.regexp_arg);
  names->SetSubMatch(args_info.submatch_arg);

  // Create geometry reader
  rtk::OraGeometryReader::Pointer oraReader = rtk::OraGeometryReader::New();
  oraReader->SetProjectionsFileNames(names->GetFileNames());
  TRY_AND_EXIT_ON_ITK_EXCEPTION( oraReader->UpdateOutputData() )

  // Write
  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter = rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject( oraReader->GetGeometry() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( xmlWriter->WriteFile() )

  return EXIT_SUCCESS;
}
