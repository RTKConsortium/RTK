# Copyright (C) 1999-2009  Free Software Foundation, Inc.
#  
# This file is part of GNU gengetopt 
#
# GNU gengetopt is free software; you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation; either version 3, or (at your option) 
# any later version. 
#
# GNU gengetopt is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details. 
#
# You should have received a copy of the GNU General Public License along 
# with gengetopt; see the file COPYING. If not, write to the Free Software 
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA. 

SUBDIRS = includes skels tests

#INCLUDES = -I@top_srcdir@/src/skels

SUFFIXES = .text .c .ggo

.text.o:
	$(TXTC) $<

bin_PROGRAMS = gengetopt

noinst_LTLIBRARIES = libgengetopt.la

libgengetopt_la_SOURCES = parser.cc scanner.cc \
globals.cpp \
argsdef.c cmdline.ggo gm.cc \
yyerror.cc gm_utils.cpp fileutils.cpp acceptedvalues.cpp ggos.cpp

gengetopt_SOURCES = gengetopt.cc

scanner.cc: $(srcdir)/scanner.ll
	$(LEX) $(LFLAGS) -o $(srcdir)/scanner.cc $(srcdir)/scanner.ll

parser.cc: $(srcdir)/parser.yy
	$(YACC) -o $(srcdir)/parser.cc $(srcdir)/parser.yy --defines=$(srcdir)/parser.h

#gengetopt_LDADD = @LIBOBJS@ @LEXLIB@ skels/libgen.a

# these are for gnulib
AM_CPPFLAGS = -I$(top_srcdir)/gl -I$(top_builddir)/gl

libgengetopt_la_LIBADD = $(top_builddir)/gl/libgnu.la \
	@LTLIBOBJS@ \
	skels/libgen.la

LDADD = $(top_builddir)/src/libgengetopt.la

EXTRA_DIST = parser.h argsdef.h gengetopt.h ggos.h gm.h gnugetopt.h \
cmdline.c cmdline.h my_sstream.h my_string.h my_map.h \
global_opts.h \
ggo_options.h \
groups.h \
parser.yy scanner.ll \
globals.h \
$(pkgdata_DATA)

# version of gengetopt linked to the ElectricFence library
# don't use it anymore: use valgrind
#EXTRA_PROGRAMS = gengetopt-efence
#gengetopt_efence_SOURCES = $(gengetopt_SOURCES)
#gengetopt_efence_LDADD = -lefence $(gengetopt_LDADD)

PROGNAME = $(top_builddir)/src/gengetopt$(EXEEXT)

# to have flex generate a case insensitive scanner
AM_LFLAGS = -i

# it may happen, usually if the sources have been downloaded from CVS
# repository, that cmdline.ggo is more recent than cmdline.c, but
# cmdline.c cannot be re-generated, even because gengetopt has
# to be built first. For instance if you only change spaces in cmdline.ggo
# cmdline.c will not be different, and thus it is not updated in the CVS
# repository; then when you make a checkout on another place, make
# would try to build cmdline.c, but to do this it would need gengetopt
# which is not built yet; so we simply touch cmdline.c (and this is safe)

.ggo.c:
	if test -f $(PROGNAME); then \
	$(PROGNAME) --input=$< --output-dir=$(srcdir) ; \
	else touch $(srcdir)/$@ ; fi

cmdline.h: cmdline.c

BUILT_SOURCES = cmdline.h cmdline.c parser.cc scanner.cc

# automatically generate gengetopt_strdup() from the replacement function
# strdup().
# gengetopt_strdup.text: strdup.c
#	echo "/* gengetopt_strdup(): automatically generated from strdup.c. */" > $@
#	cat strdup.c | sed -e 's/^strdup (/gengetopt_&/' \
#               -e 's/^char \* *$$/&/' \
#               -e '/^#include.*$$/D' -e '/^$$/D' >> $@

# we distribute these files because who uses gengetopt might need them
# these are installed in $(prefix)/share/gengetopt
pkgdata_DATA = gnugetopt.h getopt.c getopt1.c

# yywrap.c is added only if it is not found on lex (flex) lib, and will be
# added automatically by automake

# automake (maybe due to a bug) doesn't added these files well,
# so we explicitely declare them
MAINTAINERCLEANFILES = scanner.cc parser.h parser.cc 

noinst_HEADERS = acceptedvalues.h errorcodes.h fileutils.h \
	gm_utils.h yyerror.h

