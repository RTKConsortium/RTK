# Copyright (C) 1999-2008 Free Software Foundation, Inc.
#
# This file is free software; as a special exception the author gives
# unlimited permission to copy and/or distribute it, with or without
# modifications, as long as this notice is preserved.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY, to the extent permitted by law; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# if gengen is not installed we simply ignore the changes

SUFFIXES = .h_skel

if NO_GENGEN
#.h_skel.cc:
#	echo "Not regenerating $@ since gengen is not installed"
# better not to use touch, otherwise we will create an empty file in
# the build directory
#	touch $@

else
GENERATE = $(GENGEN)

.h_skel.cc:
	name="`echo $* | sed 's/^.*\///g'`"; \
	echo "$$name"; \
	$(GENERATE) -i $< -F $*.h -f $$name --separate-files --expand-tabs --output-dir=$(srcdir) --force
endif

INCLUDES = -I@top_srcdir@/src

noinst_LTLIBRARIES = libgen.la

libgen_la_SOURCES = $(BUILT_SOURCES)

BUILT_SOURCES = header.h header.cc c_source.h c_source.cc \
handle_help.h handle_version.h handle_help.cc handle_version.cc \
generic_option.h required_option.h dependant_option.h \
generic_option.cc required_option.cc dependant_option.cc \
group_counter.h group_option.h \
group_counter.cc group_option.cc \
print_help_string.h print_help_string.cc \
multiple_opt_list.cc multiple_opt_list.h \
multiple_fill_array.cc multiple_fill_array.h \
copyright.cc copyright.h \
free_string.cc free_string.h \
free_multiple.cc free_multiple.h \
reset_group.cc reset_group.h \
exit_failure.cc exit_failure.h \
update_given.cc update_given.h \
option_arg.cc option_arg.h \
given_field.cc given_field.h \
clear_given.cc clear_given.h \
clear_arg.cc clear_arg.h \
free_list.cc free_list.h \
file_save.cc file_save.h \
file_save_multiple.cc file_save_multiple.h \
init_args_info.cc init_args_info.h \
custom_getopt_gen.cc custom_getopt_gen.h \
check_modes.cc check_modes.h \
enum_decl.cc enum_decl.h

EXTRA_DIST = header.h_skel c_source.h_skel handle_help.h_skel \
handle_version.h_skel generic_option.h_skel \
required_option.h_skel dependant_option.h_skel \
group_counter.h_skel group_option.h_skel \
print_help_string.h_skel \
multiple_opt_list.h_skel \
multiple_fill_array.h_skel \
copyright.h_skel free_string.h_skel \
free_multiple.h_skel \
reset_group.h_skel \
exit_failure.h_skel \
update_given.h_skel \
option_arg.h_skel \
given_field.h_skel \
clear_given.h_skel \
clear_arg.h_skel \
free_list.h_skel \
file_save.h_skel \
file_save_multiple.h_skel \
init_args_info.h_skel \
custom_getopt_gen.h_skel \
check_modes.h_skel \
enum_decl.h_skel \
$(BUILT_SOURCES)

built-clean:
	cd @srcdir@ && rm -f $(BUILT_SOURCES)