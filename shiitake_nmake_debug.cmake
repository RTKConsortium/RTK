# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "shiitake.clb")
set(CTEST_BUILD_NAME "Windows7-64bit-NMake-Debug")
set(CTEST_BUILD_CONFIGURATION Debug)
set(CTEST_CMAKE_GENERATOR "NMake Makefiles")
set(CTEST_MAKE_PROGRAM "nmake")
set(CTEST_GIT_COMMAND "C:/cygwin/bin/git")
set(dashboard_binary_name "RTK_win64_nmake_debug")
set( ENV{ITK_DIR} "Z:/src/itk/win64_nmake_debug")

set( ENV{Path} "C:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/Bin/amd64;C:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/vcpackages;C:/Program Files (x86)/Microsoft Visual Studio 9.0/Common7/IDE;C:/Program Files/Microsoft SDKs/Windows/v7.0/Bin/x64;C:/Program Files/Microsoft SDKs/Windows/v7.0/Bin;C:/Windows/Microsoft.NET/Framework64/v3.5;C:/Windows/Microsoft.NET/Framework/v3.5;C:/Windows/Microsoft.NET/Framework64/v2.0.50727;C:/Windows/Microsoft.NET/Framework/v2.0.50727;C:/Program Files/Microsoft SDKs/Windows/v7.0/Setup;C:/Windows/system32;C:/Windows;C:/Windows/System32/Wbem;C:/Windows/System32/WindowsPowerShell/v1.0/;C:/Program Files (x86)/CMake 2.8/bin")
set( ENV{Lib} "C:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/Lib/amd64;C:/Program Files/Microsoft SDKs/Windows/v7.0/Lib/X64;" )
set( ENV{Include} "C:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/Include;C:/Program Files/Microsoft SDKs/Windows/v7.0/Include;C:/Program Files/Microsoft SDKs/Windows/v7.0/Include/gl;" )
set( ENV{CPU} "AMD64" )
set( ENV{APPVER} "6.1" )
set( FLAGS_DEBUG "/D_DEBUG /MTd /Zi  /Ob0 /Od /RTC1" )
set(CONFIGURE_OPTIONS
  -DCMAKE_CXX_FLAGS_DEBUG=${FLAGS_DEBUG}
  -DCMAKE_C_FLAGS_DEBUG=${FLAGS_DEBUG}
)
include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

