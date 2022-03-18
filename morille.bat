call "c:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
ctest -R "(rtk|RTK)" -VV -S morille_build_rtk_in_itk.cmake
ctest -R "(rtk|RTK)" -VV -S morille_tbb.cmake
ctest -VV -S morille_build_itk.cmake
ctest -VV -S morille.cmake
REM C:\Windows\System32\shutdown.exe /s
