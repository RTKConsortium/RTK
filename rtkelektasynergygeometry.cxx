#include "rtkThreeDCircularGeometryXMLFile.h"
#include "rtkelektasynergygeometry_ggo.h"

#include "rtkDbf.h"

#include <itkTimeProbe.h>

std::string GetImageIDFromDicomUID(const args_info_rtkelektasynergygeometry &args_info)
{
  // Open image database file
  rtk::DbfFile dbImage(args_info.image_db_arg);
  if (!dbImage.is_open()){
    std::cerr << "Couldn't open " << args_info.image_db_arg << std::endl;
    exit(EXIT_FAILURE);
  }

  // Search for correct record
  bool bReadOk;
  do {
    bReadOk = dbImage.ReadNextRecord();
  }
  while(bReadOk && std::string(args_info.dicom_uid_arg) != dbImage.GetFieldAsString("DICOM_UID"));

  // Error message if not found
  if(!bReadOk)
    {
    std::cerr << "Couldn't find acquisition with DICOM_UID " << args_info.dicom_uid_arg
              << " in table " << args_info.image_db_arg << std::endl;
    exit(EXIT_FAILURE);
    }

  return dbImage.GetFieldAsString("DBID");
}

void GetProjInfoFromDB(const std::string &imageID,
                       const args_info_rtkelektasynergygeometry &args_info,
                       std::vector<float> &projAngle,
                       std::vector<float> &projFlexX,
                       std::vector<float> &projFlexY)
{
  // Open frame database file
  rtk::DbfFile dbFrame(args_info.frame_db_arg);
  if (!dbFrame.is_open()){
    std::cerr << "Couldn't open " << args_info.frame_db_arg << std::endl;
    exit(EXIT_FAILURE);
  }

  // Go through the database, select correct records and get data
  while( dbFrame.ReadNextRecord() )
    {
    if(dbFrame.GetFieldAsString("IMA_DBID") == imageID)
      {
      projAngle.push_back(dbFrame.GetFieldAsDouble("PROJ_ANG"));
      projFlexX.push_back(dbFrame.GetFieldAsDouble("U_CENTRE"));
      projFlexY.push_back(dbFrame.GetFieldAsDouble("V_CENTRE"));
      }
    }
}

int main(int argc, char * argv[])
{
  GGO(rtkelektasynergygeometry, args_info);

  // RTK geometry object
  typedef rtk::ThreeDCircularGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();

  // Get information from Synergy database
  std::vector<float> projAngle, projFlexX, projFlexY;
  GetProjInfoFromDB(GetImageIDFromDicomUID(args_info), args_info, projAngle, projFlexX, projFlexY);

  // Global parameters
  geometry->SetSourceToDetectorDistance(args_info.sdd_arg);
  geometry->SetSourceToIsocenterDistance(args_info.sid_arg);

  // Projection matrices
  for(unsigned int noProj=0; noProj<projAngle.size(); noProj++)
    {
    geometry->AddProjection(projAngle[noProj], projFlexX[noProj], projFlexY[noProj]);
    }

  // Write
  rtk::ThreeDCircularGeometryXMLFileWriter::Pointer xmlWriter = rtk::ThreeDCircularGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject(&(*geometry));
  xmlWriter->WriteFile();

  return EXIT_SUCCESS;
}
