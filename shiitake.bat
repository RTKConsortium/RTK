ctest -R "(rtk|RTK)" -VV -S shiitake_build_rtk_in_itk.cmake
ctest -VV -S shiitake_build_itk.cmake
ctest -VV -S shiitake.cmake
REM C:\Windows\System32\shutdown.exe /s
