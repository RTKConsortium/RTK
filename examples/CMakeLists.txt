SET(RTK_DIR ${RTK_BINARY_DIR})

ADD_SUBDIRECTORY(HelloWorld)
ADD_SUBDIRECTORY(FirstReconstruction)

# Testing of the examples
if(BUILD_TESTING)
  add_test(HelloWorldTest ${EXECUTABLE_OUTPUT_PATH}/HelloWorld)
  set_tests_properties (HelloWorldTest
    PROPERTIES 
    PASS_REGULAR_EXPRESSION "RTK Hello World!"
    )

  add_test(FirstReconstructionTest ${EXECUTABLE_OUTPUT_PATH}/FirstReconstruction)
  add_test(FirstReconstructionCheckTest ${EXECUTABLE_OUTPUT_PATH}/rtkcheckimagequality output.mha)
  set_tests_properties(FirstReconstructionCheckTest PROPERTIES DEPENDS FirstReconstructionTest)

endif()

