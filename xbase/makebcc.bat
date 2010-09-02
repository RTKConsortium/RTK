
rem This batch file builds the xbase library using Borland C++ 5.5


del *.bak
del *.obj

bcc32 -c -I.. dbf.cpp       > compout
bcc32 -c -I.. database.cpp >> compout
bcc32 -c -I.. exp.cpp      >> compout
bcc32 -c -I.. expfunc.cpp  >> compout
bcc32 -c -I.. expproc.cpp  >> compout
bcc32 -c -I.. fields.cpp   >> compout
bcc32 -c -I.. html.cpp     >> compout
bcc32 -c -I.. index.cpp    >> compout
bcc32 -c -I.. lock.cpp     >> compout
bcc32 -c -I.. stack.cpp    >> compout
bcc32 -c -I.. xbase.cpp    >> compout
bcc32 -c -I.. xbexcept.cpp >> compout
bcc32 -c -I.. memo.cpp     >> compout
bcc32 -c -I.. xdate.cpp    >> compout
bcc32 -c -I.. xbstring.cpp >> compout
bcc32 -c -I.. xbfilter.cpp >> compout
bcc32 -c -I.. ndx.cpp      >> compout
bcc32 -c -I.. ntx.cpp      >> compout

del xbase.lib
tlib xbase.lib /C +dbf.obj +database.obj +exp.obj +expfunc.obj >> compout
tlib xbase.lib /C +expproc.obj +fields.obj +html.obj +index.obj >> compout
tlib xbase.lib /C +lock.obj +stack.obj +xbase.obj +xbexcept.obj >> compout
tlib xbase.lib /C +memo.obj +xdate.obj +xbstring.obj +xbfilter.obj >> compout
tlib xbase.lib /C +ndx.obj +ntx.obj