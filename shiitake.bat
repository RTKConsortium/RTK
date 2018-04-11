ctest -VV -S shiitake_static_release.cmake
ctest -VV -S shiitake_shared_release.cmake
ctest -VV -S shiitake_static_release_simplertk.cmake
ctest -VV -S shiitake_shared_release_simplertk.cmake
ctest -VV -S shiitake_static_release_nofftw.cmake
ctest -VV -S shiitake_static_debug_nofftw.cmake
ctest -VV -S shiitake_static_debug.cmake
ctest -VV -S shiitake_shared_debug.cmake
REM C:\Windows\System32\shutdown.exe /s