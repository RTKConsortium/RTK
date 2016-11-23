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

#include <srtkConditional.h>
#include <srtkCommand.h>
#include <srtkFunctionCommand.h>
#include <srtkCastImageFilter.h>

TEST( ConditionalTest, ConditionalTest1 ) {

  // a quick check to make sure the conditional works
  typedef rtk::simple::Conditional<true, int, float>::Type IntType;
  typedef rtk::simple::Conditional<false, int, float>::Type FloatType;


  EXPECT_EQ ( typeid( IntType ).name(), typeid( int ).name() );
  EXPECT_EQ ( typeid( FloatType ).name(), typeid( float ).name() );

  return;

}

TEST( ProcessObject, GlobalWarning ) {
  // Basic coverage test of setting and getting. Need separate
  // specific check for propagation of warning to ITK.

  namespace srtk = rtk::simple;

  EXPECT_TRUE( srtk::ProcessObject::GetGlobalWarningDisplay() );

  srtk::ProcessObject::GlobalWarningDisplayOff();
  EXPECT_FALSE( srtk::ProcessObject::GetGlobalWarningDisplay() );

  srtk::ProcessObject::GlobalWarningDisplayOn();
  EXPECT_TRUE( srtk::ProcessObject::GetGlobalWarningDisplay() );

  srtk::ProcessObject::SetGlobalWarningDisplay(false);
  EXPECT_FALSE( srtk::ProcessObject::GetGlobalWarningDisplay() );

}


TEST( ProcessObject, Command_Register ) {
  // Test the references between Process Objects and command.
  // Try to be mean and break stuff

  namespace srtk = rtk::simple;

  // Case 0a: stack,  command first
  {
  srtk::Command cmd;
  srtk::CastImageFilter po1;
  po1.AddCommand(srtk::srtkAnyEvent, cmd);
  EXPECT_TRUE(po1.HasCommand(srtk::srtkAnyEvent));
  }

  // Case 0b: stack, process first
  {
  srtk::CastImageFilter po1;
  srtk::Command cmd;
  po1.AddCommand(srtk::srtkAnyEvent, cmd);
  EXPECT_TRUE(po1.HasCommand(srtk::srtkAnyEvent));
  }

  // Case 1a: single command, command deleted first
  {
  std::auto_ptr<srtk::CastImageFilter> po1(new srtk::CastImageFilter());
  std::auto_ptr<srtk::Command> cmd(new srtk::Command());
  po1->AddCommand(srtk::srtkAnyEvent, *cmd);

  EXPECT_TRUE(po1->HasCommand(srtk::srtkAnyEvent));
  cmd.reset();
  EXPECT_FALSE(po1->HasCommand(srtk::srtkAnyEvent));
  }

  // Case 1b: single command, process deleted first
  {
  std::auto_ptr<srtk::CastImageFilter> po1( new srtk::CastImageFilter());
  std::auto_ptr<srtk::Command> cmd(new srtk::Command());
  po1->AddCommand(srtk::srtkAnyEvent, *cmd);
  po1.reset();
  }

  // Case 2a: single command, multiple processes, command deleted first
  {
  std::auto_ptr<srtk::CastImageFilter> po1(new srtk::CastImageFilter());
  std::auto_ptr<srtk::CastImageFilter> po2(new srtk::CastImageFilter());
  std::auto_ptr<srtk::CastImageFilter> po3(new srtk::CastImageFilter());

  std::auto_ptr<srtk::Command> cmd(new srtk::Command());
  po1->AddCommand(srtk::srtkAnyEvent, *cmd);
  po2->AddCommand(srtk::srtkStartEvent, *cmd);
  po3->AddCommand(srtk::srtkEndEvent, *cmd);
  cmd.reset();
  }

  // Case 2b: single command, multiple processes, processes mostly deleted first
  {
  std::auto_ptr<srtk::CastImageFilter> po1(new srtk::CastImageFilter());
  std::auto_ptr<srtk::CastImageFilter> po2(new srtk::CastImageFilter());
  std::auto_ptr<srtk::CastImageFilter> po3(new srtk::CastImageFilter());

  std::auto_ptr<srtk::Command> cmd(new srtk::Command());
  po1->AddCommand(srtk::srtkAnyEvent, *cmd);
  po2->AddCommand(srtk::srtkStartEvent, *cmd);
  po3->AddCommand(srtk::srtkEndEvent, *cmd);

  EXPECT_TRUE(po1->HasCommand(srtk::srtkAnyEvent));
  EXPECT_TRUE(po2->HasCommand(srtk::srtkStartEvent));
  EXPECT_TRUE(po3->HasCommand(srtk::srtkEndEvent));

  po1.reset();
  EXPECT_TRUE(po2->HasCommand(srtk::srtkStartEvent));
  EXPECT_TRUE(po3->HasCommand(srtk::srtkEndEvent));
  po2.reset();
  EXPECT_TRUE(po3->HasCommand(srtk::srtkEndEvent));
  cmd.reset();
  EXPECT_FALSE(po3->HasCommand(srtk::srtkEndEvent));
  }

  // Case 3a: multiple commands, command deleted mostly first
  {
  std::auto_ptr<srtk::CastImageFilter> po1(new srtk::CastImageFilter());
  std::auto_ptr<srtk::Command> cmd1(new srtk::Command());
  std::auto_ptr<srtk::Command> cmd2(new srtk::Command());
  std::auto_ptr<srtk::Command> cmd3(new srtk::Command());

  po1->AddCommand(srtk::srtkAnyEvent, *cmd1);
  po1->AddCommand(srtk::srtkStartEvent, *cmd2);
  po1->AddCommand(srtk::srtkEndEvent, *cmd3);

  EXPECT_TRUE(po1->HasCommand(srtk::srtkAnyEvent));
  EXPECT_TRUE(po1->HasCommand(srtk::srtkStartEvent));
  EXPECT_TRUE(po1->HasCommand(srtk::srtkEndEvent));

  cmd1.reset();
  EXPECT_FALSE(po1->HasCommand(srtk::srtkAnyEvent));
  EXPECT_TRUE(po1->HasCommand(srtk::srtkStartEvent));
  EXPECT_TRUE(po1->HasCommand(srtk::srtkEndEvent));
  cmd2.reset();
  EXPECT_FALSE(po1->HasCommand(srtk::srtkAnyEvent));
  EXPECT_FALSE(po1->HasCommand(srtk::srtkStartEvent));
  EXPECT_TRUE(po1->HasCommand(srtk::srtkEndEvent));
  po1.reset();
  }

  // Case 3b: multiple commands, process object deleted first
  {
  std::auto_ptr<srtk::CastImageFilter> po1(new srtk::CastImageFilter());
  std::auto_ptr<srtk::Command> cmd1(new srtk::Command());
  std::auto_ptr<srtk::Command> cmd2(new srtk::Command());
  std::auto_ptr<srtk::Command> cmd3(new srtk::Command());
  po1->AddCommand(srtk::srtkAnyEvent, *cmd1);
  po1->AddCommand(srtk::srtkStartEvent, *cmd2);
  po1->AddCommand(srtk::srtkEndEvent, *cmd3);
  po1.reset();

  }


}

TEST( ProcessObject, Command_Add ) {
  // Add command for events and verifies the state

  namespace srtk = rtk::simple;

  srtk::CastImageFilter po1;
  srtk::Command cmd;

  // check initial state
  EXPECT_FALSE(po1.HasCommand(srtk::srtkAnyEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkAbortEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkDeleteEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkEndEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkIterationEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkProgressEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkStartEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkUserEvent));

  po1.AddCommand(srtk::srtkAnyEvent, cmd);
  EXPECT_TRUE(po1.HasCommand(srtk::srtkAnyEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkAbortEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkDeleteEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkEndEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkIterationEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkProgressEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkStartEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkUserEvent));

  po1.RemoveAllCommands();
  EXPECT_FALSE(po1.HasCommand(srtk::srtkAnyEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkAbortEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkDeleteEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkEndEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkIterationEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkProgressEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkStartEvent));
  EXPECT_FALSE(po1.HasCommand(srtk::srtkUserEvent));

  po1.AddCommand(srtk::srtkAnyEvent, cmd);
  po1.AddCommand(srtk::srtkAbortEvent, cmd);
  po1.AddCommand(srtk::srtkDeleteEvent, cmd);
  po1.AddCommand(srtk::srtkEndEvent, cmd);
  po1.AddCommand(srtk::srtkIterationEvent, cmd);
  po1.AddCommand(srtk::srtkProgressEvent, cmd);
  po1.AddCommand(srtk::srtkStartEvent, cmd);
  po1.AddCommand(srtk::srtkUserEvent, cmd);

  EXPECT_TRUE(po1.HasCommand(srtk::srtkAnyEvent));
  EXPECT_TRUE(po1.HasCommand(srtk::srtkAbortEvent));
  EXPECT_TRUE(po1.HasCommand(srtk::srtkDeleteEvent));
  EXPECT_TRUE(po1.HasCommand(srtk::srtkEndEvent));
  EXPECT_TRUE(po1.HasCommand(srtk::srtkIterationEvent));
  EXPECT_TRUE(po1.HasCommand(srtk::srtkProgressEvent));
  EXPECT_TRUE(po1.HasCommand(srtk::srtkStartEvent));
  EXPECT_TRUE(po1.HasCommand(srtk::srtkUserEvent));
}

TEST( ProcessObjectDeathTest, DeleteCommandActiveProcess )
{
  // if a command is deleted while the process is active, it is
  // expected for the program to terminate.
  namespace srtk = rtk::simple;

  class DeleteCommandAtCommand
  : public ProcessObjectCommand
  {
  public:
    DeleteCommandAtCommand(rtk::simple::ProcessObject &po, float abortAt, Command *cmd)
      : ProcessObjectCommand(po),
        m_AbortAt(abortAt),
        m_Cmd(cmd)
      {
      }

    virtual void Execute( )
      {
        if ( m_Process.GetProgress() >= m_AbortAt )
          delete m_Cmd;
      }

    float m_AbortAt;
    Command *m_Cmd;
};

  srtk::CastImageFilter po;
  srtk::Image img(100,100,100, srtk::srtkUInt16);

  srtk::Command *cmd1 = new srtk::Command();
  DeleteCommandAtCommand cmd2(po, .01, cmd1);

  po.AddCommand(srtk::srtkAnyEvent, *cmd1);
  po.AddCommand(srtk::srtkProgressEvent, cmd2);


  po.SetNumberOfThreads(1);
  ::testing::FLAGS_gtest_death_test_style = "fast";

  ASSERT_DEATH(po.Execute(img), "Cannot delete Command during execution");

}


TEST( Event, Test1 )
{
  // Test print of EventEnum with output operator
  namespace srtk = rtk::simple;

  std::stringstream ss;
  ss << srtk::srtkAnyEvent;
  EXPECT_EQ("AnyEvent", ss.str());
  ss.str("");
  ss << srtk::srtkAbortEvent;
  EXPECT_EQ("AbortEvent", ss.str());
  ss.str("");
  ss << srtk::srtkDeleteEvent;
  EXPECT_EQ("DeleteEvent", ss.str());
  ss.str("");
  ss << srtk::srtkEndEvent;
  EXPECT_EQ("EndEvent", ss.str());
  ss.str("");
  ss << srtk::srtkIterationEvent;
  EXPECT_EQ("IterationEvent", ss.str());
  ss.str("");
  ss << srtk::srtkProgressEvent;
  EXPECT_EQ("ProgressEvent", ss.str());
  ss.str("");
  ss << srtk::srtkStartEvent;
  EXPECT_EQ("StartEvent", ss.str());
  ss.str("");
  ss << srtk::srtkUserEvent;
  EXPECT_EQ("UserEvent", ss.str());
}


TEST( Command, Test1 ) {
  // Basic test.
  namespace srtk = rtk::simple;

  srtk::Command cmd1;
  // not copy construct able
  //srtk::Command cmd2(cmd1);

  // not assignable
  //cmd1 = cmd1;

  // Does nothing
  cmd1.Execute();

  EXPECT_EQ( "Command", cmd1.GetName() );
  cmd1.SetName("SomeName");
  EXPECT_EQ( "SomeName", cmd1.GetName() );

}


TEST( ProcessObject, Command_Ownership ) {
  // Test the functionality of the ProcessObject Owning the Command
  namespace srtk = rtk::simple;

  static int destroyedCount = 0;

  class HeapCommand
    : public srtk::Command
  {
  public:
    HeapCommand() : v(false) {};
    ~HeapCommand() {++destroyedCount;}
    virtual void Execute() {v=true;}
    using Command::SetOwnedByProcessObjects;
    using Command::GetOwnedByProcessObjects;
    using Command::OwnedByProcessObjectsOn;
    using Command::OwnedByProcessObjectsOff;

    bool v;
  };

  {
  // test set/get/on/off
  HeapCommand cmd;
  EXPECT_FALSE(cmd.GetOwnedByProcessObjects());
  cmd.SetOwnedByProcessObjects(true);
  EXPECT_TRUE(cmd.GetOwnedByProcessObjects());
  cmd.OwnedByProcessObjectsOff();
  EXPECT_FALSE(cmd.GetOwnedByProcessObjects());
  cmd.OwnedByProcessObjectsOn();
  EXPECT_TRUE(cmd.GetOwnedByProcessObjects());

  HeapCommand *cmd1 = new HeapCommand();
  cmd1->OwnedByProcessObjectsOn();
  EXPECT_EQ(0,destroyedCount);
  delete cmd1;
  EXPECT_EQ(1,destroyedCount);
  }
  EXPECT_EQ(2,destroyedCount);

  // case 1
  // single po, multiple cmds
  {
  srtk::CastImageFilter po;
  srtk::Image img(5,5, srtk::srtkUInt16);

  HeapCommand *cmd2 = new HeapCommand();
  cmd2->OwnedByProcessObjectsOn();
  po.AddCommand(srtk::srtkAnyEvent, *cmd2);

  EXPECT_FALSE(cmd2->v);
  EXPECT_NO_THROW( po.Execute(img) );
  EXPECT_TRUE(cmd2->v);
  cmd2->v = false;

  HeapCommand *cmd3 = new HeapCommand();
  cmd3->OwnedByProcessObjectsOn();
  po.AddCommand(srtk::srtkAnyEvent, *cmd3);

  EXPECT_FALSE(cmd2->v);
  EXPECT_FALSE(cmd3->v);
  EXPECT_NO_THROW( po.Execute(img) );
  EXPECT_TRUE(cmd2->v);
  EXPECT_TRUE(cmd3->v);
  cmd2->v = false;

  delete cmd3;
  EXPECT_EQ(3,destroyedCount);
  }
  EXPECT_EQ(4,destroyedCount);

  // case 2
  // cmd registered to multiple PO
  {
  std::auto_ptr<srtk::CastImageFilter> po1(new srtk::CastImageFilter());
  std::auto_ptr<srtk::CastImageFilter> po2(new srtk::CastImageFilter());

  HeapCommand *cmd = new HeapCommand();
  cmd->OwnedByProcessObjectsOn();

  po1->AddCommand(srtk::srtkAnyEvent, *cmd);
  po1->AddCommand(srtk::srtkStartEvent, *cmd);

  EXPECT_TRUE(po1->HasCommand(srtk::srtkAnyEvent));
  EXPECT_TRUE(po1->HasCommand(srtk::srtkStartEvent));

  po2->AddCommand(srtk::srtkAnyEvent, *cmd);
  EXPECT_TRUE(po2->HasCommand(srtk::srtkAnyEvent));


  po2.reset();

  EXPECT_TRUE(po1->HasCommand(srtk::srtkAnyEvent));
  EXPECT_TRUE(po1->HasCommand(srtk::srtkStartEvent));
  EXPECT_EQ(4,destroyedCount);
  }
  EXPECT_EQ(5,destroyedCount);


}

TEST( Command, Test2 ) {
  // Check basic name functionality
  namespace srtk = rtk::simple;

  srtk::Command cmd1;


}

TEST( FunctionCommand, Test1 ) {
  // Basic test.
  namespace srtk = rtk::simple;

  srtk::FunctionCommand cmd1;
  // not copy construct able
  //srtk::Command cmd2(cmd1);

  // not assignable
  //cmd1 = cmd1;

  // Does nothing
  cmd1.Execute();

  EXPECT_EQ( "FunctionCommand", cmd1.GetName() );
  cmd1.SetName("AnotherName");
  EXPECT_EQ( "AnotherName", cmd1.GetName() );
}

namespace
{

struct MemberFunctionCommandTest
{
  MemberFunctionCommandTest():v(0){}

  void DoSomething() {v=99;}
  int v;
};

int gValue = 0;
void functionCommand(void)
{
  gValue = 98;
}

void functionCommandWithClientData(void *_data)
{
  int &data = *reinterpret_cast<int*>(_data);
  data = 97;
}

}

TEST( FunctionCommand, Test2 ) {
  // check execution of different callbacks types
  namespace srtk = rtk::simple;

  MemberFunctionCommandTest mfct;
  EXPECT_EQ(0,mfct.v);

  srtk::FunctionCommand cmd1;
  cmd1.SetCallbackFunction( &mfct, &MemberFunctionCommandTest::DoSomething );
  cmd1.Execute();
  EXPECT_EQ(99,mfct.v);
  mfct.v = 0;

  EXPECT_EQ(0,gValue);
  cmd1.SetCallbackFunction(functionCommand);
  cmd1.Execute();
  EXPECT_EQ(98,gValue);
  EXPECT_EQ(0,mfct.v);


  cmd1.SetCallbackFunction(functionCommandWithClientData, &gValue);
  cmd1.Execute();
  EXPECT_EQ(97,gValue);
  EXPECT_EQ(0,mfct.v);
}
