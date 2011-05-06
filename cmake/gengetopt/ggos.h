/*
This file is licensed to you under the license specified in the included file
`LICENSE'. Look there for further details.
*/


#ifndef _GENGETOPT_GGOS_H
#define _GENGETOPT_GGOS_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <list>
#include <iostream>

#include "acceptedvalues.h"
#include "argsdef.h"

/**
 * The structure for storing an option specified in the .ggo file
 */
struct gengetopt_option
{
  char short_opt; /**< the short option specification (one char) */
  char * long_opt; /**< the short option specification */
  char * desc; /**< the option description */
  int type; /**< the type of the option (possible values in `argsdef.h') */
  int flagstat ; /**< if the option is of type flag, this indicates its state (on/off) */
  int required; /**< whether the option required */
  bool required_set; /**< whether the required property was set */
  char * var_arg; /**< canonized long_opt + "_arg" = argument var */
  int default_given ; /**< if a default is given */
  char * default_string ; /**< default value for this option, if string */
  char * group_value; /**< group name, if it's part of an option group */
  char * group_desc; /**< group description, if it's part of an option group */
  char * mode_value; /**< mode name, if it's part of an option mode */
  char * mode_desc; /**< mode description, if it's part of an option mode */
  bool multiple; /**< whether this option can be given more than once */
  char * multiple_min; /**< minimum occurrences of a multiple option (-1: not specified) */
  char * multiple_max; /**< maximum occurrences of a multiple option (-1: not specified) */
  bool arg_is_optional; /**< whether the argument is optional */
  bool hidden; /**< whether this option will be hidden from the help output */
  char *type_str; /**< Alternative name for type,
                     e.g. "URL" or "SECONDS" */
  const AcceptedValues *acceptedvalues; /**< values that can be passed to this option */
  char *section; /**< the section of this option */
  char *section_desc; /**< the description associated with the possible section */
  char *dependon; /**< the name of the option this one depends on */

  char *text_before; /**< a possible text specified before this option */
  char *text_after; /**< a possible text specified after this option */
  
  char *details; /**< possible further details for this option that will be
  printed only if --detailed-help is specified */

  /**< parser information */
  char *filename; /**< source file */
  int linenum; /**< line number */

  gengetopt_option();
};

/** the list storing gengetopt options */
typedef std::list<gengetopt_option *> gengetopt_option_list;

std::ostream & operator <<(std::ostream &s, gengetopt_option &opt);

#endif /* _GENGETOPT_GGOS_H */
