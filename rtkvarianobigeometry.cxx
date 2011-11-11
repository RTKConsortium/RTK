#include "rtkvarianobigeometry_ggo.h"
#include "rtkGgoFunctions.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkProjectionsReader.h"
#include "itkVarianObiXMLFileReader.h"

#include <itkRegularExpressionSeriesFileNames.h>
#include <itkTimeProbe.h>

int main(int argc, char * argv[])
{
  GGO(rtkvarianobigeometry, args_info);

  // RTK geometry object
  typedef itk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();

  // Read Varian XML file (for common geometric information)
  itk::VarianObiXMLFileReader::Pointer obiXmlReader;
  obiXmlReader = itk::VarianObiXMLFileReader::New();
  obiXmlReader->SetFilename(args_info.xml_file_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( obiXmlReader->GenerateOutputInformation() )

  // Generate file names of projections
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(false);
  names->SetRegularExpression(args_info.regexp_arg);
  names->SetSubMatch(0);

  // Constants used to generate projection matrices
  itk::MetaDataDictionary &dic = *(obiXmlReader->GetOutputObject() );
  typedef itk::MetaDataObject< double > MetaDataDoubleType;
  const double sdd = dynamic_cast<MetaDataDoubleType *>(dic["CalibratedSID"].GetPointer() )->GetMetaDataObjectValue();
  const double sid = dynamic_cast<MetaDataDoubleType *>(dic["CalibratedSAD"].GetPointer() )->GetMetaDataObjectValue();

  typedef itk::MetaDataObject< std::string > MetaDataStringType;
  double offsetx;
  std::string fanType = dynamic_cast<const MetaDataStringType *>(dic["FanType"].GetPointer() )->GetMetaDataObjectValue();
  if(itksys::SystemTools::Strucmp(fanType.c_str(), "HalfFan") == 0)
    {
    // Half Fan (offset detector), get lateral offset from XML file
    offsetx =
      dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetX"].GetPointer() )->GetMetaDataObjectValue() +
	   dynamic_cast<MetaDataDoubleType *>(dic["DetectorPosLat"].GetPointer() )->GetMetaDataObjectValue();
    }
  else
    {
    // Full Fan (centered detector)
    offsetx =
      dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetX"].GetPointer() )->GetMetaDataObjectValue();
    }
  const double offsety =
    dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetY"].GetPointer() )->GetMetaDataObjectValue();

  // Projection matrices
  for(unsigned int noProj=0; noProj<names->GetFileNames().size(); noProj++)
    {
    typedef unsigned int                    InputPixelType;
    typedef itk::Image< InputPixelType, 2 > InputImageType;

    // Projections reader (for angle)
    itk::HndImageIOFactory::RegisterOneFactory();

    typedef itk::ImageFileReader< InputImageType > ReaderType;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( names->GetFileNames()[noProj] );
    TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->UpdateOutputInformation() )

    const double angle =
      dynamic_cast<MetaDataDoubleType *>(reader->GetMetaDataDictionary()["dCTProjectionAngle"].GetPointer())->GetMetaDataObjectValue();

    geometry->AddProjection(sid, sdd, angle, -1*offsetx, -1*offsety);
    }

  // Write
  itk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter;
  xmlWriter = itk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject(&(*geometry) );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( xmlWriter->WriteFile() )

  return EXIT_SUCCESS;
}
