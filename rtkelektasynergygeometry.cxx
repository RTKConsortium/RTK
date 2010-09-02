#include "rtkThreeDCircularGeometryXMLFile.h"
#include "rtkelektasynergygeometry_ggo.h"

#include "xbase/xbase.h"

#include <itkTimeProbe.h>

//Synergy DB constants
#define NO_FIELD_ID_IMAGE_IN_IMAGE    short(0)
#define NO_FIELD_DICOM_UID_IN_IMAGE   short(27)
#define NO_FIELD_ID_IMAGE_IN_FRAME    short(1)
#define NO_FIELD_NUM_FRAME            short(3)
#define NO_FIELD_ANGLE_IN_FRAME       short(6)
#define NO_FIELD_X_FLEX_IN_FRAME      short(7)
#define NO_FIELD_Y_FLEX_IN_FRAME      short(8)

std::string GetImageIDFromDicomUID(const args_info_rtkelektasynergygeometry &args_info)
{
  //Open image database
  xbXBase db;

  xbShort rc;
  xbDbf imageTable(&db);
  rc = imageTable.OpenDatabase( args_info.image_db_arg );
  if(rc!=XB_NO_ERROR){
    std::cerr << "Couldn't open " << args_info.image_db_arg << std::endl;
    exit(EXIT_FAILURE);
  }
  rc = imageTable.GetFirstRecord();
  if(rc!=XB_NO_ERROR)
    {
    std::cerr << "Couldn't open first record of " << args_info.image_db_arg << std::endl;
    exit(EXIT_FAILURE);
    }
  std::string currentDicomUidImage;
  do
    currentDicomUidImage = imageTable.GetField(NO_FIELD_DICOM_UID_IN_IMAGE);
  while(currentDicomUidImage!=std::string(args_info.dicom_uid_arg) && imageTable.GetNextRecord() == XB_NO_ERROR);
  if(currentDicomUidImage!=std::string(args_info.dicom_uid_arg))
    {
    std::cerr << "Couldn't file acquisition with DICOM_UID " << args_info.dicom_uid_arg
              << " in table " << args_info.image_db_arg << std::endl;
    exit(EXIT_FAILURE);
    }
    currentDicomUidImage = imageTable.GetField(NO_FIELD_DICOM_UID_IN_IMAGE);


  const std::string imageID(imageTable.GetField(NO_FIELD_ID_IMAGE_IN_IMAGE));
  imageTable.CloseDatabase();
  return imageID;
}

void GetProjInfoFromDB(const std::string &imageID,
                       const args_info_rtkelektasynergygeometry &args_info,
                       std::vector<float> &projAngle,
                       std::vector<float> &projFlexX,
                       std::vector<float> &projFlexY)
{
  //Open image database
  xbXBase db;

  xbShort rc;
  xbDbf frameTable(&db);
  rc = frameTable.OpenDatabase( args_info.frame_db_arg );
  if(rc!=XB_NO_ERROR){
    std::cerr << "Couldn't open " << args_info.frame_db_arg << std::endl;
    exit(EXIT_FAILURE);
  }
  rc = frameTable.GetFirstRecord();
  if(rc!=XB_NO_ERROR)
    {
    std::cerr << "Couldn't open first record of " << args_info.frame_db_arg << std::endl;
    exit(EXIT_FAILURE);
    }

  do
    if(frameTable.GetField(NO_FIELD_ID_IMAGE_IN_FRAME) == imageID)
      {
      projAngle.push_back(frameTable.GetFloatField(NO_FIELD_ANGLE_IN_FRAME));
      projFlexX.push_back(frameTable.GetFloatField(NO_FIELD_X_FLEX_IN_FRAME));
      projFlexY.push_back(frameTable.GetFloatField(NO_FIELD_Y_FLEX_IN_FRAME));
      }
  while(frameTable.GetNextRecord() == XB_NO_ERROR);
  frameTable.CloseDatabase();
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
