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

#include "rtkimagxgeometry_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkImagXGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

int main(int argc, char * argv[])
{
  GGO(rtkimagxgeometry, args_info);

  // Image Type
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Create geometry reader
  rtk::ImagXGeometryReader<OutputImageType>::Pointer imagxReader = rtk::ImagXGeometryReader<OutputImageType>::New();
  imagxReader->SetProjectionsFileNames( rtk::GetProjectionsFileNamesFromGgo(args_info) );
  if (args_info.dicomcalibration_flag)
    {
    imagxReader->SetReadCalibrationFromProjections(true);
    }
  else
    {
    if (! (args_info.calibration_given)&&(args_info.room_setup_given))
        itkGenericExceptionMacro("Calibration and room setup information required, either from projection's DICOM information or from external xml files");

    imagxReader->SetReadCalibrationFromProjections(false);
    imagxReader->SetCalibrationXMLFileName(args_info.calibration_arg);
    imagxReader->SetRoomXMLFileName(args_info.room_setup_arg);
    }
  imagxReader->SetDetectorOffset(args_info.offset_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( imagxReader->UpdateOutputData() )

  // Write
  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter = rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject( imagxReader->GetGeometry() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( xmlWriter->WriteFile() )

  return EXIT_SUCCESS;
}
