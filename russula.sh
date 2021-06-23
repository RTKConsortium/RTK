#!/bin/bash -x
export CXXFLAGS="-fPIC -std=c++11"
export CUDA_BIN_PATH=/home/srit/Downloads/cuda102/bin
export CUDA_LIB_PATH=/home/srit/Downloads/cuda102/lib64
export LD_LIBRARY_PATH=$CUDA_LIB_PATH:$LD_LIBRARY_PATH
export PATH=/home/srit/Downloads/cmake-3.20.4-linux-x86_64/bin:$PATH

ctest -R "(rtk|RTK)" -S /home/srit/src/rtk/rtk-dashboard/russula_build_rtk_in_itk.cmake -V
ctest -R "(rtk|RTK)" -S /home/srit/src/rtk/rtk-dashboard/russula_tbb.cmake -V
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_suse_gcc.cmake -V
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_suse_gcc_shared.cmake -V
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_suse_gcc_debug.cmake -V
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_style.cmake -V
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_doxygen.cmake -V
rsync -e 'ssh -i /home/srit/.ssh/nophrase' -a --delete \
    /home/srit/src/rtk/dashboard_tests/RTK-Doxygen/Doxygen/html \
    ssh.creatis.insa-lyon.fr:/home/srit/src/rtk/dashboard_tests/RTK-Doxygen/Doxygen
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_suse_gcc_nocuda_valgrind.cmake -V
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_suse_gcc_cuda_valgrind.cmake -V
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_suse_gcc_cuda_cov.cmake -V
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_suse_gcc_space.cmake -V

