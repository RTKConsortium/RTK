cmake_minimum_required(VERSION 2.8)
project(SimpleRTKExamples)

# Add individual cxx executables
add_executable ( RTKFirstReconstruction RTKFirstReconstruction.cxx )
target_link_libraries ( RTKFirstReconstruction ${SimpleRTK_LIBRARIES} )

