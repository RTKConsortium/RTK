/**
 * Copyright (C) 1999-2007  Free Software Foundation, Inc.
 *
 * This file is part of GNU gengetopt
 *
 * GNU gengetopt is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU gengetopt is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with gengetopt; see the file COPYING. If not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

#ifndef _GENGETOPT_H
#define _GENGETOPT_H

#include "acceptedvalues.h"
#include "ggos.h"

int gengetopt_define_package (char * s) ;
int gengetopt_define_version (char * s) ;
int gengetopt_define_purpose (char * s) ;
int gengetopt_define_description (char * s) ;
int gengetopt_define_usage (char * s) ;

/**
 * Sets the "global" section specification that will be then set
 * in the following options
 * @param s The section
 * @param sd The section description
 */
void gengetopt_set_section (const char * s, const char *sd) ;

/**
 * Sets the "global" text string that will be inserted in
 * a specific field of the next option
 * @param desc The text string
 */
void gengetopt_set_text (const char * desc) ;

/**
 * Sets the "global" text string containing the arguments
 * that complement the command line arguments of gengetopt.
 * @param args
 */
void gengetopt_set_args (const char *args);

int gengetopt_add_group (const char * s, const char *desc, int required) ;
int gengetopt_add_mode (const char * s, const char *desc) ;

int gengetopt_has_option (const char * long_opt, char short_opt);
int gengetopt_add_option (const char * long_opt, char short_opt,
                          const char * desc,
                          int type, int flagstat, int required,
                          const char *default_value,
                          const char * group_value,
                          const char * mode_value,
                          const char * type_str,
                          const AcceptedValues *acceptedvalues,
                          int multiple = 0,
                          int argoptional = 0);

int gengetopt_has_option (gengetopt_option *opt);
int gengetopt_check_option (gengetopt_option *opt,
    bool groupoption = false, bool modeoption = false);
int gengetopt_add_option (gengetopt_option *opt);

#endif /* _GENGETOPT_H */
