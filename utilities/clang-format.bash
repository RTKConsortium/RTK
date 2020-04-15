set -e
set -x
RTK_DIR=$(dirname $0)/..

ITK_SOURCE_DIR=$(grep ITK_SOURCE_DIR ${ITK_DIR}/CMakeCache.txt | sed "s/.*=//g")
if ! test -e ${RTK_DIR}/../../../CMake/itkVersion.cmake
then
    echo "$(basename $0) must be run from the ITK source directory"
fi
for i in \
  ${RTK_DIR}/cmake/*.cxx \
  ${RTK_DIR}/include/*.h* \
  ${RTK_DIR}/src/*.c{xx,u} \
  ${RTK_DIR}/applications/*/*.{h,cxx,hxx} \
  ${RTK_DIR}/examples/FirstReconstruction/*.cxx \
  ${RTK_DIR}/utilities/ITKCudaCommon/include/*.h* \
  ${RTK_DIR}/utilities/ITKCudaCommon/src/*.cxx \
  ${RTK_DIR}/test/*{h,cxx}
do
	clang-format -style=file -i $i
done

