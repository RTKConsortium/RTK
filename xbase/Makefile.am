# $Id: Makefile.am,v 1.13 2003/08/20 01:53:27 gkunkel Exp $
#
#	This file is part of the XBase libraries
#    Copyright (C) 1998 Denis Pershin (dyp@inetlab.com)
#
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#    Contact:
#
#      Mail:
#
#        Technology Associates, Inc.
#        XBase Project
#        1455 Deming Way #11
#        Sparks, NV 89434
#        USA
#
#      Email:
#
#        xbase@techass.com
#
#      See our website at:
#
#        xdb.sourceforge.net
#

INCLUDES = -I$(topdir)

lib_LTLIBRARIES = libxbase.la

pkginclude_HEADERS = dbf.h exp.h html.h ndx.h retcodes.h xbase.h \
	xdate.h xstack.h xtypes.h xbexcept.h xbstring.h index.h ntx.h   \
        xbconfig.h xbfilter.h

#install-data-hook:
#	(cd $(includedir); rm -f xbase.h; ln -s xbase/xbase.h xbase.h)

libxbase_la_SOURCES = dbf.cpp exp.cpp expfunc.cpp expproc.cpp fields.cpp \
	html.cpp lock.cpp memo.cpp ndx.cpp stack.cpp xbase.cpp xdate.cpp \
	xbexcept.cpp xbstring.cpp index.cpp ntx.cpp xbfilter.cpp 

EXTRA_DIST = \
	xbase.h \
	xbconfigw32.h \
	makefile.g95 makebcc.bat \
	xbase.ide


libxbase_la_LDFLAGS = -version-info 1:0:0 
#libxbase_la_LDFLAGS = -version-info 0:0:0 -release @RELEASE@
libxbase_la_LIBADD = 

MAINTAINERCLEANFILES = Makefile.in stamp-h.in xbconfig.h.in
