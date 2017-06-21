#!/usr/bin/env bash


# Updates Swig .i documentation files from SimpleRTK's Doxygen XML.
#
# To generate the input to this script, the SimpleRTK Doxygen must be built.
#
# Configuration for directories need to be manually done in the
# config_vars.sh file. The RTK Doxygen XML will automatically be
# downloaded if needed.
#
# Usage: SWigDocUpdate.sh
#

die() {
    echo "$@" 1>&2
    exit 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load configuration variable from file
. ${DIR}/config_vars.sh || die 'Unable to find local \"config_vars.sh\" configuration file.'


[ -e ${SRTK_BUILD_DIR}/Documentation/xml/sitkImage_8h.xml ] ||
   die "Uable to find SimpleITK Doxygen XML! SimpleITK Doxygen needs to be generated with the Documentation target!"

${PYTHON_EXECUTABLE} doxyall.py ${SRTK_BUILD_DIR}/Documentation/xml/ ${SimpleRTK}/Wrapping/Python/PythonDocstrings.i
${PYTHON_EXECUTABLE} doxyall.py -j ${SRTK_BUILD_DIR}/Documentation/xml/ ${SimpleRTK}/Wrapping/Java/JavaDoc.i

if [ ! -d ${SimpleRTK}/Wrapping/R/Packaging/SimpleITK/man/ ] ; then
    mkdir -p ${SimpleRTK}/Wrapping/R/Packaging/SimpleITK/man/
fi
${PYTHON_EXECUTABLE} doxyall.py -r ${SRTK_BUILD_DIR}/Documentation/xml/ ${SimpleRTK}/Wrapping/R/Packaging/SimpleRTK/man/
