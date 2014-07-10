#!/bin/bash -x

if ! test -e /tmp/itk
then
	git clone git://itk.org/ITK.git /tmp/itk
fi
cd /tmp/itk
BUILDTYPE=Release
if test $# -gt 3
then
	git checkout v${1}.${2}.${3}
	DIRNAME=itk${1}${2}${3}-${4}
	BUILDTYPE=${4}
elif test $# -gt 2
then
	git checkout v${1}.${2}.${3}
	DIRNAME=itk${1}${2}${3}
elif test $# -gt 1
then
	git checkout v${1}.${2}
	DIRNAME=itk${1}${2}${3}
else
	git checkout master
	DIRNAME=itkHEAD
fi

mkdir -p /tmp/$DIRNAME/lin64-dg
cd /tmp/$DIRNAME/lin64-dg
cmake /tmp/itk -DCMAKE_BUILD_TYPE=${BUILDTYPE} -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DModule_ITKReview=ON -DITK_USE_FFTWD=ON -DITK_USE_FFTWF=ON -DITK_USE_SYSTEM_FFTW=ON -DCMAKE_INSTALL_PREFIX=/home/srit/src/$DIRNAME/lin64-dg
make -j18 install
cd
rm -fr /tmp/$DIRNAME

