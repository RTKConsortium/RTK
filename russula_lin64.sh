#!/bin/bash -x
export CXX="/usr/bin/c++ -fPIC -std=c++11"
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_suse_gcc.cmake -V
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_suse_gcc_debug.cmake -V
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_style.cmake -V
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_doxygen.cmake -V
rsync -e 'ssh -i /home/srit/.ssh/nophrase' -a --delete \
    /home/srit/src/rtk/dashboard_tests/RTK-Doxygen/Doxygen/html \
    ssh.creatis.insa-lyon.fr:/home/srit/src/rtk/dashboard_tests/RTK-Doxygen/Doxygen
export PATH=/home/srit/src/cmake/lin64_gcc_472/bin:$PATH
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_suse_gcc472_cuda41_itk4.cmake -V

