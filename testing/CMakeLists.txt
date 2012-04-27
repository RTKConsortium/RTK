
SET(REFERENCE_TEST /home/mvila/RTKnew/RTK/testing/rtkthreedphantomreferencetest.cxx)
ADD_RTK_EXECUTABLE(rtkthreedphantomreferencetest ${REFERENCE_TEST})

ADD_TEST(rtkthreedphantomreferencetest ${EXECUTABLE_OUTPUT_PATH}/rtkthreedphantomreferencetest --phantomfile ${EXECUTABLE_OUTPUT_PATH}/Phantom_Conf.xml)

