#!/bin/bash -x

if ! test -e /tmp/itk
then
    git clone git://itk.org/ITK.git /tmp/itk
fi
mkdir -p /tmp/itk${1}${2}${3}/lin64-dg
cd /tmp/itk
git checkout v${1}.${2}.${3}
cd /tmp/itk${1}${2}${3}/lin64-dg
cmake /tmp/itk -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DModule_ITKReview=ON -DITK_USE_FFTWD=ON -DITK_USE_FFTWF=ON -DITK_USE_SYSTEM_FFTW=ON -DCMAKE_INSTALL_PREFIX=/home/srit/src/itk${1}${2}${3}/lin64-dg
make -j18 install
cd
rm -fr /tmp/itk${1}${2}${3}

