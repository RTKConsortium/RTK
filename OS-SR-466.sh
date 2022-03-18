#!/bin/bash -x
export CXXFLAGS="-fPIC -std=c++11"
/Applications/CMake.app/Contents/bin/ctest -R "(rtk|RTK|FirstReconstruction)" -S /Users/srit/src/rtk/rtk-dashboard/OS-SR-466_build_rtk_in_itk.cmake -V

