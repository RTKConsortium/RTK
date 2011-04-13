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
  obiXmlReader->GenerateOutputInformation();

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
  const double offsetx =
    dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetX"].GetPointer() )->GetMetaDataObjectValue();
  const double offsety =
    dynamic_cast<MetaDataDoubleType *>(dic["CalibratedDetectorOffsetY"].GetPointer() )->GetMetaDataObjectValue();

  // Global parameters
  geometry->SetSourceToDetectorDistance(sdd);
  geometry->SetSourceToIsocenterDistance(sid);

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
    reader->UpdateOutputInformation();

    const double angle =
      dynamic_cast<MetaDataDoubleType *>(reader->GetMetaDataDictionary()["dCTProjectionAngle"].GetPointer())->GetMetaDataObjectValue();

    geometry->AddProjection(angle, -1*offsetx, -1*offsety);
    }

  // Write
  itk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter;
  xmlWriter = itk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject(&(*geometry) );
  xmlWriter->WriteFile();

  return EXIT_SUCCESS;
}
