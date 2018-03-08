if test $# -gt 0
then
	export CC=/home/srit/src/gcc/gcc${1}${2}${3}-install/bin/gcc
	export CXX=/home/srit/src/gcc/gcc${1}${2}${3}-install/bin/c++
	export LD_LIBRARY_PATH=/home/srit/src/gcc/gcc${1}${2}${3}-install/lib64:$LD_LIBRARY_PATH 
	export LD_LIBRARY_PATH=/home/srit/src/gcc/gcc${1}${2}${3}-install/lib:$LD_LIBRARY_PATH 
	export PATH=/home/srit/src/gcc/gcc${1}${2}${3}-install/bin:$PATH 
	mkdir /home/srit/src/itk/lin64_gcc_${1}${2}${3}
	cd /home/srit/src/itk/lin64_gcc_${1}${2}${3}/
else
	mkdir /home/srit/src/itk/lin64-dg
	cd /home/srit/src/itk/lin64-dg
fi
cmake ../itk -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DModule_ITKReview=ON -DITK_USE_FFTWD=ON -DITK_USE_FFTWF=ON -DITK_USE_SYSTEM_FFTW=ON
make -j12

