/*=========================================================================
*
*  Copyright Insight Software Consortium & RTK Consortium
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
#include <SimpleRTKTestHarness.h>

#include <srtkImage.h>
#include <srtkImageFileReader.h>
#include <srtkImageFileWriter.h>
#include <srtkHashImageFilter.h>
#include <srtkCastImageFilter.h>
#include <srtkPixelIDValues.h>
#include <srtkStatisticsImageFilter.h>
#include <srtkCommand.h>

#include "itkVectorImage.h"

TEST(BasicFilters,ImageFilter) {
  namespace srtk = rtk::simple;

  srtk::CastImageFilter caster;
  srtk::ImageFilter<1> &filter = caster;

  filter.DebugOn();
}

TEST(BasicFilters,ProcessObject_Debug) {
  namespace srtk = rtk::simple;

  srtk::CastImageFilter caster;
  srtk::ProcessObject &filter = caster;

  EXPECT_FALSE(filter.GetGlobalDefaultDebug());
  EXPECT_FALSE(filter.GetDebug());

  filter.DebugOff();
  EXPECT_FALSE(filter.GetDebug());
  EXPECT_FALSE(filter.GetGlobalDefaultDebug());

  filter.DebugOn();
  EXPECT_TRUE(filter.GetDebug());
  EXPECT_FALSE(filter.GetGlobalDefaultDebug());

  filter.GlobalDefaultDebugOn();
  EXPECT_TRUE(filter.GetDebug());
  EXPECT_TRUE(filter.GetGlobalDefaultDebug());

  filter.GlobalDefaultDebugOff();
  EXPECT_TRUE(filter.GetDebug());
  EXPECT_FALSE(filter.GetGlobalDefaultDebug());

  filter.GlobalDefaultDebugOn();

  srtk::CastImageFilter caster2;
  EXPECT_TRUE(caster2.GetDebug());
  EXPECT_TRUE(caster2.GetGlobalDefaultDebug());

}

TEST(BasicFilters,ProcessObject_NumberOfThreads) {
  namespace srtk = rtk::simple;

  srtk::CastImageFilter caster;
  srtk::ProcessObject &filter = caster;

  unsigned int gNum = filter.GetGlobalDefaultNumberOfThreads();
  EXPECT_NE(filter.GetGlobalDefaultNumberOfThreads(), 0u);
  EXPECT_NE(filter.GetNumberOfThreads(), 0u);
  EXPECT_EQ(filter.GetNumberOfThreads(), filter.GetGlobalDefaultNumberOfThreads());

  filter.SetNumberOfThreads(3);
  EXPECT_EQ(3u, filter.GetNumberOfThreads());
  EXPECT_EQ(gNum, filter.GetGlobalDefaultNumberOfThreads());

  filter.SetGlobalDefaultNumberOfThreads(gNum+1);
  EXPECT_EQ(gNum+1, filter.GetGlobalDefaultNumberOfThreads());
  EXPECT_EQ(3u, filter.GetNumberOfThreads());

  srtk::CastImageFilter caster2;
  EXPECT_EQ(gNum+1, caster2.GetNumberOfThreads());
  EXPECT_EQ(gNum+1, caster2.GetGlobalDefaultNumberOfThreads());
}

TEST(BasicFilters,Cast) {
  rtk::simple::HashImageFilter hasher;
  rtk::simple::ImageFileReader reader;

  reader.SetFileName ( dataFinder.GetFile ( "Input/RA-Float.nrrd" ) );
  rtk::simple::Image image = reader.Execute();
  ASSERT_TRUE ( image.GetITKBase() != NULL );
  hasher.SetHashFunction ( rtk::simple::HashImageFilter::MD5 );
  EXPECT_EQ ( "3ccccde44efaa3d688a86e94335c1f16", hasher.Execute ( image ) );

  EXPECT_EQ ( image.GetPixelIDValue(), rtk::simple::srtkFloat32 );
  EXPECT_EQ ( image.GetPixelID(), rtk::simple::srtkFloat32 );
  EXPECT_EQ ( image.GetPixelIDTypeAsString(), "32-bit float" );

  typedef std::map<std::string,rtk::simple::PixelIDValueType> MapType;
  MapType mapping;
  mapping["2f27e9260baeba84fb83dd35de23fa2d"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkUInt8;
  mapping["2f27e9260baeba84fb83dd35de23fa2d"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkInt8;
  mapping["a963bd6a755b853103a2d195e01a50d3"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkUInt16;
  mapping["a963bd6a755b853103a2d195e01a50d3"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkInt16;
  mapping["6ceea0011178a955b5be2d545d107199"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkUInt32;
  mapping["6ceea0011178a955b5be2d545d107199"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkInt32;
  mapping["efa4c3b27349b97b02a64f3d2b5ca9ed"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkUInt64;
  mapping["efa4c3b27349b97b02a64f3d2b5ca9ed"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkInt64;
  mapping["3ccccde44efaa3d688a86e94335c1f16"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkFloat32;
  mapping["ac0228acc17038fd1f1ed28eb2841c73"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkFloat64;
  mapping["226dabda8fc07f20e2b9e44ca1c83955"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkComplexFloat32;
  mapping["e92cbb187a92610068d7de0cb23364db"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkComplexFloat64;
  mapping["2f27e9260baeba84fb83dd35de23fa2d"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkVectorUInt8;
  mapping["2f27e9260baeba84fb83dd35de23fa2d"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkVectorInt8;
  mapping["a963bd6a755b853103a2d195e01a50d3"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkVectorUInt16;
  mapping["a963bd6a755b853103a2d195e01a50d3"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkVectorInt16;
  mapping["6ceea0011178a955b5be2d545d107199"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkVectorUInt32;
  mapping["6ceea0011178a955b5be2d545d107199"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkVectorInt32;
  mapping["efa4c3b27349b97b02a64f3d2b5ca9ed"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkVectorUInt64;
  mapping["efa4c3b27349b97b02a64f3d2b5ca9ed"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkVectorInt64;
  mapping["3ccccde44efaa3d688a86e94335c1f16"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkVectorFloat32;
  mapping["ac0228acc17038fd1f1ed28eb2841c73"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkVectorFloat64;
  mapping["srtkLabelUInt8"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkLabelUInt8;
  mapping["srtkLabelUInt16"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkLabelUInt16;
  mapping["srtkLabelUInt32"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkLabelUInt32;
  mapping["srtkLabelUInt64"] = (rtk::simple::PixelIDValueType)rtk::simple::srtkLabelUInt64;

  bool failed = false;

  // Loop over the map, load each file, and compare the hash value
  for ( MapType::iterator it = mapping.begin(); it != mapping.end(); ++it )
    {
    rtk::simple::PixelIDValueType pixelID = it->second;
    std::string hash = it->first;

    std::cerr << std::flush;
    std::cerr << std::flush;
    if ( pixelID == rtk::simple::srtkUnknown )
      {
      std::cerr << "Enum value: " << pixelID << " (" << hash << ") is unknown and not instantiated" << std::endl;
      continue;
      }

    std::cerr << "Testing casting to pixelID: " << pixelID << " is " << rtk::simple::GetPixelIDValueAsString ( pixelID ) << std::endl;

    try
      {
      rtk::simple::CastImageFilter caster;
      rtk::simple::Image test = caster.SetOutputPixelType ( pixelID ).Execute ( image );

      hasher.SetHashFunction ( rtk::simple::HashImageFilter::MD5 );
      EXPECT_EQ ( hash, hasher.Execute ( test ) ) << "Cast to " << rtk::simple::GetPixelIDValueAsString ( pixelID );

      }
    catch ( ::rtk::simple::GenericException &e )
      {

      // hashing currently doesn't work for label images
      if ( hash.find( "srtkLabel" ) == 0 )
        {
        std::cerr << "Hashing currently is not implemented for Label Images" << std::endl;
        }
      else
        {
        failed = true;
        std::cerr << "Failed to hash: " << e.what() << std::endl;
        }

      continue;
      }

  }
  EXPECT_FALSE ( failed ) << "Cast failed, or could not take the hash of the imoge";

}

TEST(BasicFilters,HashImageFilter) {
  rtk::simple::HashImageFilter hasher;
  EXPECT_EQ ( "rtk::simple::HashImageFilter\nHashFunction: SHA1\n", hasher.ToString() );
  EXPECT_EQ ( rtk::simple::HashImageFilter::SHA1, hasher.SetHashFunction ( rtk::simple::HashImageFilter::SHA1 ).GetHashFunction() );
  EXPECT_EQ ( rtk::simple::HashImageFilter::MD5, hasher.SetHashFunction ( rtk::simple::HashImageFilter::MD5 ).GetHashFunction() );
}

TEST(BasicFilters,Cast_Commands) {
  // test cast filter with a bunch of commands

  namespace srtk = rtk::simple;
  srtk::Image img = srtk::ReadImage( dataFinder.GetFile ( "Input/RA-Short.nrrd" ) );
  EXPECT_EQ ( "a963bd6a755b853103a2d195e01a50d3", srtk::Hash(img, srtk::HashImageFilter::MD5));

  srtk::CastImageFilter caster;
  caster.SetOutputPixelType( srtk::srtkInt32 );

  ProgressUpdate progressCmd(caster);
  caster.AddCommand(srtk::srtkProgressEvent, progressCmd);

  CountCommand abortCmd(caster);
  caster.AddCommand(srtk::srtkAbortEvent, abortCmd);

  CountCommand deleteCmd(caster);
  caster.AddCommand(srtk::srtkDeleteEvent, deleteCmd);

  CountCommand endCmd(caster);
  caster.AddCommand(srtk::srtkEndEvent, endCmd);

  CountCommand iterCmd(caster);
  caster.AddCommand(srtk::srtkIterationEvent, iterCmd);

  CountCommand startCmd(caster);
  caster.AddCommand(srtk::srtkStartEvent, startCmd);

  CountCommand userCmd(caster);
  caster.AddCommand(srtk::srtkUserEvent, userCmd);


  srtk::Image out = caster.Execute(img);
  EXPECT_EQ ( "6ceea0011178a955b5be2d545d107199", srtk::Hash(out, srtk::HashImageFilter::MD5));

  EXPECT_EQ ( 1.0f, caster.GetProgress() );
  EXPECT_EQ ( 1.0f, progressCmd.m_Progress );
  EXPECT_EQ ( 0, abortCmd.m_Count );
  EXPECT_EQ ( 1, deleteCmd.m_Count );
  EXPECT_EQ ( 1, endCmd.m_Count );
  EXPECT_EQ ( 0, iterCmd.m_Count );
  EXPECT_EQ ( 1, startCmd.m_Count );
  EXPECT_EQ ( 0, userCmd.m_Count );

}
