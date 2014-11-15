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
#include "srtkMacro.h"

#include "SimpleRTKTestHarness.h"

static const char * DESCRIPTION = "We expect this exception";

class srtkExceptionsTest
  : public ::testing::Test
{
public:

  void ThrowsrtkException( void )
  {
    srtkExceptionMacro( << DESCRIPTION );
  }
};

TEST_F(srtkExceptionsTest, Test1) {
  ASSERT_THROW( ThrowsrtkException(), ::rtk::simple::GenericException );

  try
    {
    ThrowsrtkException();
    }
  catch ( ::rtk::simple::GenericException &e )
    {
    // could do some nifty testing here too
    EXPECT_EQ ( e.GetNameOfClass(), std::string("GenericException") );
    //EXPECT_NE ( std::string ( e.GetLocation() ),  "" ); HACK FIXME
    // exception revision
    return;
    }

  // should gotten that exception
  FAIL();
}

TEST_F(srtkExceptionsTest, Test2) {

  // this can only be tested when true, if it was false the file won't compile
  srtkStaticAssert( true, "this is just a test" );

  SUCCEED();
}

TEST_F(srtkExceptionsTest, Test3) {

  // This test is designed to improve coverage of the GenericException class

  // Default constructor
  const rtk::simple::GenericException empty;
  rtk::simple::GenericException e0;


  rtk::simple::GenericException e1( __FILE__, __LINE__ );

  rtk::simple::GenericException e2( __FILE__, __LINE__, "testing yet another constructor" );

  // copy constructor
  rtk::simple::GenericException e3( e2 );


  // asignment
  e0 = e2;
  e0 = e1;
  e0 = empty;

  // test self assigment too
  e0 = e0;


  EXPECT_TRUE( e1 == e1 );
  EXPECT_TRUE( empty == empty );
  EXPECT_FALSE( e2 == e1 );

  EXPECT_NO_THROW( e2.ToString() );
  EXPECT_NO_THROW( e2.GetLocation() );
  EXPECT_NO_THROW( e2.GetDescription() );
  EXPECT_NO_THROW( e2.GetFile() );
  EXPECT_NO_THROW( e2.GetLine() );
  EXPECT_NO_THROW( e2.what() );

  // check accessor for empty/null
  EXPECT_NO_THROW( empty.ToString() );
  EXPECT_NO_THROW( empty.GetLocation() );
  EXPECT_NO_THROW( empty.GetDescription() );
  EXPECT_NO_THROW( empty.GetFile() );
  EXPECT_NO_THROW( empty.GetLine() );
  EXPECT_NO_THROW( empty.what() );

}
