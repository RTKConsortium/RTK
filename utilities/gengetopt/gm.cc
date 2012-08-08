/**
 * Copyright (C) 1999-2010  Free Software Foundation, Inc.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <set>
#include <algorithm> // for pair

#include <fstream>

extern "C"
{
#include "argsdef.h"
#include "global_opts.h"
};

#include "ggo_options.h"

#include "gm.h"
#include "my_sstream.h"

#include "groups.h"
#include "skels/option_arg.h"
#include "skels/required_option.h"
#include "skels/dependant_option.h"
#include "skels/generic_option.h"
#include "skels/group_option.h"
#include "skels/group_counter.h"
#include "skels/handle_help.h"
#include "skels/handle_version.h"
#include "skels/print_help_string.h"
#include "skels/multiple_opt_list.h"
#include "skels/multiple_fill_array.h"
#include "skels/free_string.h"
#include "skels/free_multiple.h"
#include "skels/reset_group.h"
#include "skels/exit_failure.h"
#include "skels/update_given.h"
#include "skels/given_field.h"
#include "skels/clear_given.h"
#include "skels/clear_arg.h"
#include "skels/free_list.h"
#include "skels/file_save.h"
#include "skels/file_save_multiple.h"
#include "skels/init_args_info.h"
#include "skels/custom_getopt_gen.h"
#include "skels/check_modes.h"
#include "skels/enum_decl.h"
#include "gm_utils.h"
#include "fileutils.h"

#ifndef FIX_UNUSED
#define FIX_UNUSED(X) (void) (X)
#endif // FIX_UNUSED

#define MAX_STARTING_COLUMN 32

#define EXE_NAME "argv[0]"

#define PARSER_NAME_PREFIX (c_source_gen_class::parser_name + "_")
#define OPTION_VALUES_NAME(n) (PARSER_NAME_PREFIX + n + "_values")

using std::endl;
using std::set;

extern char * gengetopt_package;
extern char * gengetopt_version;
extern char * gengetopt_purpose;
extern char * gengetopt_description;
extern char * gengetopt_usage;
extern char * gengetopt_input_filename;

extern groups_collection_t gengetopt_groups;
extern modes_collection_t gengetopt_modes;

// a map where for each mode we store the corresponding given field names
// and the options
typedef std::pair<string, string> OptionValueElem;
typedef std::list<OptionValueElem> ModeOptions;
typedef std::map<string, ModeOptions> ModeOptionMap;

static ModeOptionMap modeOptionMap;

static const ModeOptionMap &getModeOptionMap() {
    if (modeOptionMap.size() == 0) {
        // it's the first time, so we build it
        struct gengetopt_option * opt;
        foropt {
            if (opt->mode_value) {
                modeOptionMap[opt->mode_value].push_back
                (std::make_pair("args_info->" + string(opt->var_arg) + "_given",
                        string("\"--") + opt->long_opt + "\""));
            }
        }
    }

    return modeOptionMap;
}

// a map associating to a mode the list of gengetopt_options
typedef std::map<string, gengetopt_option_list> ModeOptMap;

static ModeOptMap modeOptMap;

static const ModeOptMap &getModeOptMap() {
    if (modeOptMap.size() == 0) {
        // it's the first time, so we build it
        struct gengetopt_option * opt;
        foropt {
            if (opt->mode_value) {
                modeOptMap[opt->mode_value].push_back(opt);
            }
        }
    }

    return modeOptMap;
}

static void _generate_option_arg(ostream &stream,
                                 unsigned int indent,
                                 struct gengetopt_option * opt);

static void
generate_help_desc_print(ostream &stream,
                         unsigned int desc_column,
                         const char *descript, const char *defval,
                         const string &values,
                         const string &show_required_string);

CmdlineParserCreator::CmdlineParserCreator (char *function_name,
                                            char *struct_name,
                                            char *unamed_options_,
                                            char *filename_,
                                            char *header_ext, char *c_ext,
                                            bool long_help_,
                                            bool no_handle_help_,
                                            bool no_help_,
                                            bool no_handle_version_,
                                            bool no_version_,
                                            bool no_handle_error_,
                                            bool conf_parser_,
                                            bool string_parser_,
                                            bool gen_version,
                                            bool gen_getopt,
                                            bool no_options_,
                                            const string &comment_,
                                            const string &outdir,
                                            const string &header_outdir,
                                            const string &src_outdir,
                                            const string &show_required) :
  filename (filename_),
  args_info_name (struct_name),
  output_dir (outdir),
  header_output_dir (header_outdir),
  src_output_dir (src_outdir),
  comment (comment_),
  unamed_options (unamed_options_),
  show_required_string (show_required),
  long_help (long_help_), no_handle_help (no_handle_help_),
  no_help (no_help_),
  no_handle_version (no_handle_version_),
  no_version (no_version_),
  no_handle_error (no_handle_error_),
  conf_parser (conf_parser_), string_parser (string_parser_),
  gen_gengetopt_version (gen_version),
  tab_indentation (0)
{
  parser_function_name = canonize_names (function_name);
  c_filename = create_filename (filename, c_ext);
  header_filename = create_filename (filename, header_ext);

  // header_gen_class
  const string stripped_header_file_name = strip_path (filename);
  set_header_file_name (stripped_header_file_name);
  header_gen_class::set_header_file_ext (header_ext);
  c_source_gen_class::set_header_file_ext (header_ext);
  if (gen_gengetopt_version)
    header_gen_class::set_generator_version
      ("version " VERSION);
  const string my_ifndefname =
    to_upper (strip_path (stripped_header_file_name));
  set_ifndefname (canonize_names (my_ifndefname.c_str ()));
  header_gen_class::set_parser_name (parser_function_name);
  const string my_package_var_name =
    to_upper (parser_function_name) + "_PACKAGE";
  const string my_version_var_name =
    to_upper (parser_function_name) + "_VERSION";
  header_gen_class::set_package_var_name (my_package_var_name);
  c_source_gen_class::set_package_var_name (my_package_var_name);
  header_gen_class::set_version_var_name (my_version_var_name);
  c_source_gen_class::set_version_var_name (my_version_var_name);
  header_gen_class::set_args_info (args_info_name);
  c_source_gen_class::set_args_info (args_info_name);
  const string uppersand = "\"";

  // if no_options then we don't need to generate update_arg,
  // but if we need to handle help or version we still need to generate it
  set_no_options (no_options_ && !no_handle_help && !no_handle_version);

  if (gengetopt_package)
    set_package_var_val
      (uppersand + gengetopt_package + uppersand);
  else
    set_package_var_val ("PACKAGE");

  if (gengetopt_version)
    set_version_var_val
      (uppersand + gengetopt_version + uppersand);
  else
    set_version_var_val ("VERSION");

  header_gen_class::set_generate_config_parser (conf_parser);

  header_gen_class::set_generate_string_parser (string_parser);
  c_source_gen_class::set_generate_string_parser (string_parser);

  // c_source_gen_class
  set_command_line (comment);
  if (gen_gengetopt_version)
    c_source_gen_class::set_generator_version
      ("version " VERSION);
  c_source_gen_class::set_parser_name (parser_function_name);
  set_source_name (filename);

  ostringstream exit_failure_str;
  exit_failure_gen_class exit_gen;
  exit_gen.set_parser_name (c_source_gen_class::parser_name);
  exit_gen.set_handle_error (! no_handle_error);
  exit_gen.generate_exit_failure (exit_failure_str);
  set_final_exit (exit_failure_str.str ());

  set_conf_parser (conf_parser);
  set_cmd_list (conf_parser || string_parser);
  set_include_getopt (gen_getopt);

  struct gengetopt_option * opt;
  gen_strdup = (unamed_options != 0 || conf_parser || string_parser);

  if (! gen_strdup)
    {
      foropt
        if (opt->type != ARG_FLAG || opt->type != ARG_NO) {
          gen_strdup = true;
          break;
        }
    }

  set_do_generate_strdup(gen_strdup);
  set_check_possible_values(has_values());
  set_multiple_token_functions(has_multiple_options_with_type());
  set_multiple_options_with_default(has_multiple_options_with_default());
  set_multiple_options(has_multiple_options());
  set_multiple_options_string(has_multiple_options_string());
  set_multiple_options_all_string(has_multiple_options_all_string());
  set_has_typed_options(has_options_with_type());
  set_has_modes(has_options_with_mode());
  set_handle_unamed(unamed_options);
  set_check_required_options(has_required() || has_dependencies() || has_multiple_options());
  set_purpose(generate_purpose());
  set_description(generate_description());
  set_no_package((gengetopt_package == 0));
  c_source_gen_class::set_has_hidden(has_hidden_options());
  header_gen_class::set_has_hidden(c_source_gen_class::has_hidden);
  c_source_gen_class::set_has_details(has_options_with_details());
  header_gen_class::set_has_details(c_source_gen_class::has_details);

  set_has_arg_types();
}

void CmdlineParserCreator::set_has_arg_types() {
    struct gengetopt_option * opt;

    set_has_arg_flag(false);
    set_has_arg_string(false);
    set_has_arg_int(false);
    set_has_arg_short(false);
    set_has_arg_long(false);
    set_has_arg_float(false);
    set_has_arg_double(false);
    set_has_arg_longdouble(false);
    set_has_arg_longlong(false);

    foropt
    {
        switch (opt->type) {
        case ARG_NO:
            break;
        case ARG_FLAG:
            set_has_arg_flag(true);
            break;
        case ARG_STRING:
            set_has_arg_string(true);
            break;
        case ARG_INT:
            set_has_arg_int(true);
            break;
        case ARG_SHORT:
            set_has_arg_short(true);
            break;
        case ARG_LONG:
            set_has_arg_long(true);
            break;
        case ARG_FLOAT:
            set_has_arg_float(true);
            break;
        case ARG_DOUBLE:
            set_has_arg_double(true);
            break;
        case ARG_LONGDOUBLE:
            set_has_arg_longdouble(true);
            break;
        case ARG_LONGLONG:
            set_has_arg_longlong(true);
            break;
        case ARG_ENUM:
            set_has_arg_enum(true);
            break;
        default:
            fprintf (stderr, "gengetopt: bug found in %s:%d!!\n",
                    __FILE__, __LINE__);
            abort ();
        }
    }

}

void
CmdlineParserCreator::generateBreak(ostream &stream, unsigned int indent)
{
  string indent_str (indent, ' ');

  stream << endl;
  stream << indent_str;
  stream << "break;";
}

int
CmdlineParserCreator::generate ()
{
  int head_result;

  head_result = generate_header_file ();
  if (head_result)
    return head_result;

  return generate_source ();
}

int
CmdlineParserCreator::generate_header_file ()
{
  if (! gengetopt_options.size())
    {
      fprintf (stderr, "gengetopt: none option given\n");
      return 1;
    }

  /* ****************************************************** */
  /* HEADER FILE******************************************* */
  /* ****************************************************** */

    string header_file = header_filename;
    if (header_output_dir.size())
        header_file = header_output_dir + "/" + header_file;
    else if (output_dir.size())
        header_file = output_dir + "/" + header_file;

    ofstream *output_file = open_fstream
            (header_file.c_str());
    generate_header (*output_file);
    output_file->close ();
    delete output_file;

    return 0;
}

/**
 * generate the enum value from a given option
 * @param name the (canonized) name of the option
 * @param val the value of the option
 * @return the enum value string
 */
static const string from_value_to_enum(const string &name, const string &val) {
    return name + "_arg_" + canonize_enum(val);
}

void
CmdlineParserCreator::generate_enum_types(ostream &stream,
                                          unsigned int indent)
{
  struct gengetopt_option * opt;
  FIX_UNUSED (indent);

  if (has_arg_enum)
      stream << endl;

  foropt {
    // if type is enum then it should also have values (checked during parsing)
    // but it's better to check it
    if (opt->type == ARG_ENUM) {
        if (! (opt->acceptedvalues)) {
            fprintf (stderr, "gengetopt: bug found in %s:%d!!\n",
                            __FILE__, __LINE__);
            abort ();
        }
        ostringstream enum_values;
        enum_decl_gen_class enum_gen;
        enum_gen.set_var_arg(opt->var_arg);
        for (AcceptedValues::const_iterator it = opt->acceptedvalues->begin();
            it != opt->acceptedvalues->end(); ++it) {
            enum_values << ", ";
            // the first enum element is set to 0
            enum_values << from_value_to_enum(opt->var_arg, *it);
            if (it == opt->acceptedvalues->begin())
                enum_values << " = 0";

        }
        enum_gen.set_enum_values(enum_values.str());
        enum_gen.generate_enum_decl(stream);
    }
  }
}

void
CmdlineParserCreator::generate_option_arg(ostream &stream,
                                          unsigned int indent)
{
  struct gengetopt_option * opt;

  foropt {
    _generate_option_arg (stream, indent, opt);
  }
}

void
_generate_option_arg(ostream &stream,
                     unsigned int indent,
                     struct gengetopt_option *opt)
{
  option_arg_gen_class option_arg_gen;

  string type = "";
  if (opt->type)
      type = arg_types[opt->type];
  string origtype = "char *";

  if (opt->multiple) {
    type += "*";
    origtype += "*";
    option_arg_gen.set_multiple(true);
  } else {
    option_arg_gen.set_multiple(false);
  }

  option_arg_gen.set_type(type);
  option_arg_gen.set_origtype(origtype);
  option_arg_gen.set_flag_arg((opt->type == ARG_FLAG));
  option_arg_gen.set_desc(opt->desc);
  option_arg_gen.set_name(opt->var_arg);
  option_arg_gen.set_has_arg(opt->type != ARG_NO);
  option_arg_gen.set_has_enum(opt->type == ARG_ENUM);

  if (opt->default_given)
    {
      option_arg_gen.set_has_default(true);
      option_arg_gen.set_default_value(opt->default_string);
    }

  if (opt->type == ARG_FLAG)
    {
      option_arg_gen.set_default_on(opt->flagstat);
    }

  if (opt->type == ARG_LONGLONG)
    {
      // the fallback type in case longlong is not supported by the compiler
      string longtype = arg_types[ARG_LONG];
      if (opt->multiple)
          longtype += "*";

      option_arg_gen.set_long_long_arg(true);
      option_arg_gen.set_longtype(longtype);
    }

  option_arg_gen.generate_option_arg(stream, indent);
}

void
CmdlineParserCreator::generate_option_given(ostream &stream,
                                            unsigned int indent)
{
  struct gengetopt_option * opt;
  string indent_str (indent, ' ');
  bool first = true;
  given_field_gen_class given_gen;

  foropt
    {
      switch (opt->type) {
      case ARG_NO:
      case ARG_FLAG:
      case ARG_STRING:
      case ARG_INT:
      case ARG_SHORT:
      case ARG_LONG:
      case ARG_FLOAT:
      case ARG_DOUBLE:
      case ARG_LONGDOUBLE:
      case ARG_LONGLONG:
      case ARG_ENUM:
          break;
      default:
        fprintf (stderr, "gengetopt: bug found in %s:%d!!\n",
                 __FILE__, __LINE__);
        abort ();
      }
      if (! first)
        stream << indent_str;
      else
        first = false;

      given_gen.set_arg_name (opt->var_arg);
      given_gen.set_long_opt (opt->long_opt);
      given_gen.set_group (opt->multiple && opt->group_value);
      given_gen.generate_given_field (stream);
    }

  if (unamed_options)
    {
      stream << endl;
      stream << indent_str;
      stream << "char **inputs ; /**< @brief unamed options (options without names) */\n" ;
      stream << indent_str;
      stream << "unsigned inputs_num ; /**< @brief unamed options number */" ;
    }
}

void
CmdlineParserCreator::generate_option_values_decl(ostream &stream,
                                                  unsigned int indent)
{
  struct gengetopt_option * opt;
  bool first = true;
  FIX_UNUSED (indent);

  foropt
    {
      if (opt->acceptedvalues) {
        if (first) {
          first = false;
        }

        stream << "extern const char *" << OPTION_VALUES_NAME(opt->var_arg) <<
          "[];  /**< @brief Possible values for " << opt->long_opt << ". */\n";
      }
    }

  if (! first)
    stream << "\n";
}

void
CmdlineParserCreator::generate_option_values(ostream &stream,
                                             unsigned int indent)
{
  struct gengetopt_option * opt;
  bool first = true;
  FIX_UNUSED (indent);

  foropt
    {
      if (opt->acceptedvalues) {
        if (first) {
          first = false;
        }

        stream << "const char *" << OPTION_VALUES_NAME(opt->var_arg) <<
          "[] = {" << opt->acceptedvalues->toString(false) <<
          ", 0}; /*< Possible values for " << opt->long_opt << ". */\n";
      }
    }

  if (! first)
    stream << "\n";
}

static void generate_option_usage_string(gengetopt_option * opt, ostream &usage) {
    const char   *type_str;

    usage << " ";

    if (!opt->required)
        usage << "[";

    switch (opt->type) {
    case ARG_NO:
    case ARG_FLAG:
        if (opt->short_opt)
            usage << "-" << opt->short_opt << "|";
        usage << "--" << opt->long_opt;
        break;
    case ARG_INT:
    case ARG_SHORT:
    case ARG_LONG:
    case ARG_FLOAT:
    case ARG_DOUBLE:
    case ARG_LONGDOUBLE:
    case ARG_LONGLONG:
    case ARG_STRING:
    case ARG_ENUM:
        if (opt->type_str)
            type_str = opt->type_str;
        else
            type_str = arg_names[opt->type];

        if (opt->short_opt)
            usage << "-" << opt->short_opt << type_str << "|";
        usage << "--" << opt->long_opt << "=" << type_str;

        break;
    default: fprintf (stderr, "gengetopt: bug found in %s:%d!!\n",
            __FILE__, __LINE__);
    abort ();
    }

    if (!opt->required)
        usage << "]";
}

const string
CmdlineParserCreator::generate_usage_string(bool use_config_package)
{
  FIX_UNUSED (use_config_package);
  // if specified by the programmer, the usage string has the precedence
  if (gengetopt_usage) {
    return gengetopt_usage;
  }

  struct gengetopt_option * opt;
  ostringstream usage;

  // otherwise the config.h package constant will be used
  if (gengetopt_package)
    usage << gengetopt_package;

  if ( long_help ) {
      // we first generate usage strings of required options
      // handle mode options separately
      foropt
          if (opt->required && !opt->hidden && !opt->mode_value) {
              generate_option_usage_string(opt, usage);
          }

      foropt
          if (!opt->required && !opt->hidden && !opt->mode_value) {
              generate_option_usage_string(opt, usage);
          }
  } else { /* if not long help we generate it as GNU standards */
      usage << " [OPTIONS]...";
  }

  string wrapped;

  if ( unamed_options )
      usage << " [" << unamed_options << "]...";

  wrap_cstr ( wrapped, strlen("Usage: "), 2, usage.str() );

  // now deal with modes
  if (has_modes && long_help) {
      const ModeOptMap &modeOptMap = getModeOptMap();

      for (ModeOptMap::const_iterator map_it = modeOptMap.begin(); map_it != modeOptMap.end(); ++map_it) {
          string mode_line; // a mode alternative in the usage string
          gengetopt_option_list::const_iterator opt_it;
          usage.str(""); // reset the usage string buffer

          for (opt_it = map_it->second.begin(); opt_it != map_it->second.end(); ++opt_it) {
              if (((*opt_it)->required) && !((*opt_it)->hidden)) {
                  generate_option_usage_string(*opt_it, usage);
              }
          }

          for (opt_it = map_it->second.begin(); opt_it != map_it->second.end(); ++opt_it) {
              if (!((*opt_it)->required) && !((*opt_it)->hidden)) {
                  generate_option_usage_string(*opt_it, usage);
              }
          }

          wrap_cstr ( mode_line, strlen("  or : "), 2, gengetopt_package + usage.str() );
          wrapped += "\\n  or : ";
          wrapped += mode_line;
      }
  }

  return wrapped;
}

static void
generate_help_desc_print(ostream &stream,
                         unsigned int desc_column,
                         const char *descript, const char *defval,
                         const string &values,
                         const string &show_required_string)
{
  string desc;
  string desc_with_default = descript;

  if (defval || values.size()) {
      desc_with_default += "  (";

      if (values.size()) {
        desc_with_default += "possible values=";
        desc_with_default += values;
        if (defval)
          desc_with_default += " ";
      }

      if (defval) {
        desc_with_default += "default=";
        desc_with_default += defval;
      }

      desc_with_default += ")";
  }

  if (show_required_string != "")
    desc_with_default += " " + show_required_string;

  wrap_cstr ( desc, desc_column, 2, desc_with_default );

  stream << desc;
}


void
CmdlineParserCreator::generate_help_option_print_from_lists(ostream &stream,
        unsigned int indent, OptionHelpList *full_option_list,
        OptionHelpList *option_list, const std::string &target_array,
        const std::string &source_array) {
    print_help_string_gen_class print_gen;

    // the index into the help arrays
    int i = 0, full_i = 0;
    // num of help strings
    int help_num = 0;

    print_gen.set_target(target_array);
    print_gen.set_from(source_array);
    print_gen.set_shared(true);
    print_gen.set_last(false);

    OptionHelpList::const_iterator it = option_list->begin();
    OptionHelpList::const_iterator it2 = full_option_list->begin();
    // the second list is surely longer so we scan that one
    for (; it != option_list->end() && it2 != full_option_list->end(); ++it2)
    {
        if (*it == *it2) {
            // when the two strings are the same it means that's a non-hidden
            // option, so we share it with the full help array
            ostringstream converted_int;
            converted_int << i;

            // the index into the help array
            print_gen.set_index(converted_int.str());

            converted_int.str("");
            converted_int << full_i;

            // the index into the full help array
            print_gen.set_full_index(converted_int.str());
            print_gen.generate_print_help_string(stream, indent);

            ++help_num;
            ++i;
            ++it;
        }
        ++full_i;
    }

    ostringstream converted_int;
    converted_int << help_num;

    // the final 0
    print_gen.set_last(true);
    print_gen.set_index(converted_int.str());
    print_gen.generate_print_help_string(stream, indent);

    // we increment it to store the final 0
    converted_int.str("");
    converted_int << ++help_num;

    set_help_string_num(converted_int.str());

}

void
CmdlineParserCreator::generate_help_option_print(ostream &stream,
                                                 unsigned int indent)
{
    OptionHelpList *option_list = generate_help_option_list();

    if (!c_source_gen_class::has_hidden && !c_source_gen_class::has_details) {
        print_help_string_gen_class print_gen;
        print_gen.set_shared(false);

        // simple help generation
        for (OptionHelpList::const_iterator it = option_list->begin();
        it != option_list->end(); ++it)
        {
            print_gen.set_helpstring(*it);
            print_gen.generate_print_help_string(stream, indent);
        }
    } else {
        // in order to avoid generating the same help string twice, and thus
        // to save memory, in case of hidden options (or details), we try to share most
        // of the strings with the full help array
        OptionHelpList *full_option_list = generate_help_option_list(true, true);

        generate_help_option_print_from_lists
        (stream, indent, full_option_list, option_list,
                c_source_gen_class::args_info + "_help",
                (c_source_gen_class::has_details ?
                        c_source_gen_class::args_info + "_detailed_help" :
                            c_source_gen_class::args_info + "_full_help"));

        delete full_option_list;
    }

    delete option_list;
}

void
CmdlineParserCreator::generate_full_help_option_print(ostream &stream,
        unsigned int indent)
{
    // generate also hidden options
    OptionHelpList *option_list = generate_help_option_list(true);

    if (!c_source_gen_class::has_details) {
        print_help_string_gen_class print_gen;
        print_gen.set_shared(false);

        for (OptionHelpList::const_iterator it = option_list->begin();
        it != option_list->end(); ++it)
        {
            print_gen.set_helpstring(*it);
            print_gen.generate_print_help_string(stream, indent);
        }
    } else {
        // in order to avoid generating the same help string twice, and thus
        // to save memory, in case of options with details, we try to share most
        // of the strings with the full help array
        OptionHelpList *full_option_list = generate_help_option_list(true, true);

        generate_help_option_print_from_lists
        (stream, indent, full_option_list, option_list,
                c_source_gen_class::args_info + "_full_help",
                c_source_gen_class::args_info + "_detailed_help");

        delete full_option_list;
    }

    delete option_list;
}

void
CmdlineParserCreator::generate_detailed_help_option_print(ostream &stream,
        unsigned int indent)
{
    // generate also hidden options and details
    OptionHelpList *option_list = generate_help_option_list(true, true);

    print_help_string_gen_class print_gen;
    print_gen.set_shared(false);

    for (OptionHelpList::const_iterator it = option_list->begin();
         it != option_list->end(); ++it)
    {
        print_gen.set_helpstring(*it);
        print_gen.generate_print_help_string(stream, indent);
    }

    delete option_list;
}

void
CmdlineParserCreator::generate_init_args_info(ostream &stream, unsigned int indent)
{
    struct gengetopt_option * opt;
    init_args_info_gen_class init_args_info_gen;
    int i = 0;
    ostringstream index;

    string help_string = c_source_gen_class::args_info;

    if (c_source_gen_class::has_details) {
        help_string += "_detailed_help";
    } else if (c_source_gen_class::has_hidden) {
        help_string += "_full_help";
    } else {
        help_string += "_help";
    }
    init_args_info_gen.set_help_strings(help_string);

    const char *current_section = 0, *current_group = 0, *current_mode = 0;

    // we have to skip section description references (that appear in the help vector)
    foropt {
        index.str("");

        if (opt->section) {
          if (!current_section || (strcmp(current_section, opt->section) != 0)) {
            // a different section reference, skip it
            current_section = opt->section;
            ++i;

            if (opt->section_desc) {
              // section description takes another line, thus we have to skip this too
              ++i;
            }
          }
        }

        // skip group desc
        if (opt->group_value) {
            if (!current_group || strcmp(current_group, opt->group_value) != 0) {
                current_group = opt->group_value;
                ++i;
            }
        }

        // skip mode desc
        if (opt->mode_value) {
            if (!current_mode || strcmp(current_mode, opt->mode_value) != 0) {
                current_mode = opt->mode_value;
                ++i;
            }
        }

        // also skip the text before
        if (opt->text_before)
            ++i;

        index << i++;

        init_args_info_gen.set_var_arg(opt->var_arg);
        init_args_info_gen.set_num(index.str());

        if (opt->multiple) {
            init_args_info_gen.set_multiple(true);
            init_args_info_gen.set_min(opt->multiple_min);
            init_args_info_gen.set_max(opt->multiple_max);
        } else {
            init_args_info_gen.set_multiple(false);
        }

        init_args_info_gen.generate_init_args_info(stream, indent);

        // skip the details
        if (opt->details)
            ++i;

        // skip the text after
        if (opt->text_after)
            ++i;

    }
}

void CmdlineParserCreator::generate_custom_getopt(ostream &stream, unsigned int indent)
{
    custom_getopt_gen_gen_class custom_getopt;

    custom_getopt.generate_custom_getopt_gen (stream, indent);
}

const string
CmdlineParserCreator::generate_purpose()
{
  string wrapped_purpose;

  if (gengetopt_purpose != NULL)
    {
      wrap_cstr(wrapped_purpose, 0, 0, gengetopt_purpose);
    }

  return wrapped_purpose;
}

const string
CmdlineParserCreator::generate_description()
{
  string wrapped_description;

  if (gengetopt_description != NULL)
    {
      wrap_cstr(wrapped_description, 0, 0, gengetopt_description);
    }

  return wrapped_description;
}


OptionHelpList *
CmdlineParserCreator::generate_help_option_list(bool generate_hidden, bool generate_details)
{
  OptionHelpList *option_list = new OptionHelpList;

  unsigned long desc_col;
  struct gengetopt_option * opt;

  int           type_len;
  const char   *type_str;
  ostringstream stream;

  // if we want to generate details then we will also generate hidden options
  if (generate_details)
      generate_hidden = true;

  /* calculate columns */
  desc_col = 0;
  foropt {
    // if (opt->hidden && !generate_hidden)
    //    continue;
    // when computing columns, we also consider hidden_options, so that
    // the --help and --full-help will be aligned just the same
    // IMPORTANT: this is also crucial due to how the help string array
    // is built starting from the full-help string array:
    // we iterate over the two lists of options and check whether the
    // corresponding strings are the same; thus, the help strings must
    // have the same space alignments, otherwise they're not equal

    unsigned int width = 2 + 4 + 2;  // ws + "-a, " + ws

    width += strlen (opt->long_opt) + 2;  // "--"

    if ((opt->type != ARG_FLAG) &&
        (opt->type != ARG_NO))
      {
        if (opt->type_str)
          type_str = opt->type_str;
        else
          type_str = arg_names[opt->type];
        type_len = strlen(type_str);

        width += type_len + 1;        // "="

        if (opt->arg_is_optional)
          width += 2; // "[" and "]"
      }

    if (width > desc_col)
      desc_col = width;
  }

  if (desc_col > MAX_STARTING_COLUMN)
    desc_col = MAX_STARTING_COLUMN;

  /* print justified options */
  char *prev_group = 0;
  char *prev_mode = 0;
  char *curr_section = 0;
  bool first_option = true;

  foropt
    {
      // if the option is hidden, avoid to print a section containing only
      // hidden options
      if (opt->section &&
              (!curr_section || strcmp (curr_section, opt->section)) &&
              (!opt->hidden || generate_hidden))
      {
          curr_section = opt->section;

          ostringstream sec_string;

          if (! first_option)
              sec_string << "\\n";

          sec_string << opt->section << ":" ;

          string wrapped_def;
          wrap_cstr(wrapped_def, 0, 0, sec_string.str());
          option_list->push_back(wrapped_def);

          if (opt->section_desc)
          {
              string wrapped_desc ( 2, ' ');
              wrap_cstr ( wrapped_desc, 2, 0, opt->section_desc );

              option_list->push_back(wrapped_desc);
          }
      }

      if (opt->group_value &&
              (! prev_group || strcmp (opt->group_value, prev_group) != 0))
      {
          string group_string = "\\n Group: ";
          string wrapped_desc;

          if (opt->group_desc && strlen (opt->group_desc))
          {
              wrapped_desc = "\\n  ";
              wrap_cstr (wrapped_desc, 2, 0, opt->group_desc);
          }

          group_string += opt->group_value + wrapped_desc;

          option_list->push_back (group_string);

          prev_group = opt->group_value;
      }

      if (opt->mode_value &&
              (! prev_mode || strcmp (opt->mode_value, prev_mode) != 0))
      {
          string mode_string = "\\n Mode: ";
          string wrapped_desc;

          if (opt->mode_desc && strlen (opt->mode_desc))
          {
              wrapped_desc = "\\n  ";
              wrap_cstr (wrapped_desc, 2, 0, opt->mode_desc);
          }

          mode_string += opt->mode_value + wrapped_desc;

          option_list->push_back (mode_string);

          prev_mode = opt->mode_value;
      }

      // a possible description to be printed before this option
      if (opt->text_before)
      {
          string wrapped_desc;
          wrap_cstr ( wrapped_desc, 0, 0, opt->text_before);

          option_list->push_back(wrapped_desc);
      }

      if (!opt->hidden || generate_hidden) {
          first_option = false;
          const char * def_val = NULL;
          string def_str = "`";

          ostringstream option_stream;

          if (opt->type == ARG_FLAG || opt->type == ARG_NO)
          {
              def_val = NULL;

              if (opt->short_opt)
                  option_stream << "  -" << opt->short_opt << ", ";
              else
                  option_stream << "      ";

              option_stream << "--" << opt->long_opt;

              if (opt->type == ARG_FLAG)
                  def_val = opt->flagstat ? "on" : "off";
          }
          else
          {
              def_val = NULL;

              if (opt->type_str)
                  type_str = opt->type_str;
              else
                  type_str = arg_names[opt->type];

              type_len = strlen(type_str);

              if (opt->short_opt)
              {
                  option_stream << "  -" << opt->short_opt << ", ";
              }
              else
              {
                  option_stream << "      ";
              }

              bool arg_optional = opt->arg_is_optional;
              option_stream << "--" << opt->long_opt
              << (arg_optional ? "[" : "")
              << "=" << type_str
              << (arg_optional ? "]" : "");

              if (opt->default_string)
              {
                  def_str += opt->default_string;
                  def_str += "'";
                  def_val = def_str.c_str();
              }
          }

          const string &option_string = option_stream.str();
          stream << option_string;
          const char *opt_desc = opt->desc;

          if ((option_string.size() >= MAX_STARTING_COLUMN) ||
                  (desc_col <= option_string.size()))
          {
              string indent (MAX_STARTING_COLUMN, ' ');
              stream << "\\n" << indent;
          }
          else
          {
              string indent (desc_col - option_string.size(), ' ');
              stream << indent;
          }

          generate_help_desc_print(stream, desc_col, opt_desc, def_val,
                  (opt->acceptedvalues ? opt->acceptedvalues->toString() : ""),
                  (opt->required && show_required_string != "" ? show_required_string : ""));

          option_list->push_back(stream.str());
          stream.str("");
      }

      // before the text after we generate details if we need to
      if (opt->details && generate_details) {
          string wrapped_desc ( 2, ' ');
          // details are indented
          wrap_cstr ( wrapped_desc, 2, 0, opt->details);

          option_list->push_back(wrapped_desc);
      }

      // a possible description to be printed after this option
      if (opt->text_after)
      {
          string wrapped_desc;
          wrap_cstr ( wrapped_desc, 0, 0, opt->text_after);

          option_list->push_back(wrapped_desc);
      }
    }

  return option_list;
}

template <typename Collection>
void generate_counter_init(const Collection &collection, const string &name, ostream &stream, unsigned int indent)
{
    string indent_str (indent, ' ');
    typename Collection::const_iterator end = collection.end();

    for ( typename Collection::const_iterator idx = collection.begin(); idx != end; ++idx)
    {
        stream << indent_str;
        stream << ARGS_STRUCT << "->" << canonize_name (idx->first) << "_" <<
            name << "_counter = 0 ;";
        stream << endl;
    }
}

void
CmdlineParserCreator::generate_given_init(ostream &stream,
                                          unsigned int indent)
{
  struct gengetopt_option * opt;
  string indent_str (indent, ' ');
  clear_given_gen_class clear_given;
  clear_given.set_arg_struct(ARGS_STRUCT);

  /* now we initialize "given" fields */
  foropt
    {
      stream << indent_str;
      clear_given.set_var_arg(opt->var_arg);
      clear_given.set_group(opt->multiple && opt->group_value);
      clear_given.generate_clear_given(stream);
    }

  // for group counter initialization
  generate_counter_init(gengetopt_groups, "group", stream, indent);

  // for mode counter initialization
  generate_counter_init(gengetopt_modes, "mode", stream, indent);
}

void
CmdlineParserCreator::generate_reset_groups(ostream &stream, unsigned int indent)
{
  struct gengetopt_option * opt;
  string indent_str (indent, ' ');
  ostringstream body;
  reset_group_gen_class reset_group;
  clear_given_gen_class clear_given;
  clear_given.set_arg_struct(ARGS_STRUCT);

  reset_group.set_args_info (c_source_gen_class::args_info);

  groups_collection_t::const_iterator end = gengetopt_groups.end();
  for ( groups_collection_t::const_iterator idx = gengetopt_groups.begin();
        idx != end; ++idx)
    {
      body.str ("");
      bool found_option = false;
      bool multiple_arg = false;

      foropt
      {
        if (opt->group_value && strcmp(opt->group_value, idx->first.c_str()) == 0)
          {
            /* now we reset "given" fields */
            stream << indent_str;
            clear_given.set_var_arg(opt->var_arg);
            if (opt->multiple && opt->group_value)
              multiple_arg = true;
            clear_given.set_group(opt->multiple && opt->group_value);
            clear_given.generate_clear_given(body);

            free_option (opt, body, indent);
            found_option = true;
          }
      }

      if (found_option)
        {
          reset_group.set_name (canonize_name (idx->first));
          reset_group.set_body (body.str ());
          reset_group.generate_reset_group (stream);
        }
    }
}

void
CmdlineParserCreator::free_option(struct gengetopt_option *opt,
                                  ostream &stream, unsigned int indent)
{
  if (opt->type == ARG_NO)
    return;

  if (opt->type != ARG_FLAG)
    {
      if (opt->multiple)
        {
          free_multiple_gen_class free_multiple;
          free_multiple.set_has_string_type(opt->type == ARG_STRING);
          free_multiple.set_structure (ARGS_STRUCT);

          free_multiple.set_opt_var (opt->var_arg);
          free_multiple.generate_free_multiple
            (stream, indent);
        }
      else
        {
          free_string_gen_class free_string;
          free_string.set_has_string_type(opt->type == ARG_STRING);
          free_string.set_structure (ARGS_STRUCT);

          free_string.set_opt_var (opt->var_arg);
          free_string.generate_free_string (stream, indent);
        }
    }
}

void
CmdlineParserCreator::generate_list_def(ostream &stream, unsigned int indent)
{
  struct gengetopt_option * opt;
  string indent_str (indent, ' ');
  multiple_opt_list_gen_class multiple_opt_list;

  /* define linked-list structs for multiple options */
  foropt
    {
      if (opt->multiple)
        {
          if (opt->type)
            {
              stream << indent_str;
              multiple_opt_list.set_arg_name (opt->var_arg);
              multiple_opt_list.generate_multiple_opt_list (stream, indent);
              stream << endl;
            }
        }
    }
}

void
CmdlineParserCreator::generate_multiple_fill_array(ostream &stream, unsigned int indent)
{
  struct gengetopt_option * opt;
  string indent_str (indent, ' ');
  multiple_fill_array_gen_class filler;

  /* copy linked list into the array */
  foropt
    {
      if (opt->multiple && opt->type)
        {
          stream << indent_str;
          filler.set_option_var_name (opt->var_arg);
          filler.set_arg_type(arg_type_constants[opt->type]);
          filler.set_type (arg_types_names[opt->type]);
          string default_string = "0";
          if (opt->default_string) {
              if (opt->type == ARG_STRING)
                  default_string = string("\"") + opt->default_string + "\"";
              else if (opt->type == ARG_ENUM)
                  default_string = from_value_to_enum(opt->var_arg, opt->default_string);
              else
                  default_string = opt->default_string;
          }
          filler.set_default_value (default_string);

          filler.generate_multiple_fill_array (stream, indent);

          stream << endl;
        }
    }
}

void
CmdlineParserCreator::generate_update_multiple_given(ostream &stream, unsigned int indent)
{
  if (! has_multiple_options())
    return;

  string indent_str (indent, ' ');

  stream << endl;
  stream << indent_str;

  update_given_gen_class update_given_gen;
  struct gengetopt_option * opt;

  foropt
    {
      if (opt->multiple)
        {
          update_given_gen.set_option_var_name (opt->var_arg);
          update_given_gen.generate_update_given (stream, indent);
        }
    }
}

void
CmdlineParserCreator::generate_check_modes(ostream &stream, unsigned int indent)
{
    // no need to check for conflict if there's only one mode
    if (gengetopt_modes.size() < 2)
        return;

    string indent_str (indent, ' ');

    stream << endl;
    stream << indent_str;

    const ModeOptionMap &modeOptionMap = getModeOptionMap();

    check_modes_gen_class check_modes_gen;

    // now we check each mode options against every other mode options:
    // the first one with the other n-1, the second one with the other n-2, etc.
    ModeOptionMap::const_iterator map_it1, map_it2;
    for (ModeOptionMap::const_iterator map_it = modeOptionMap.begin(); map_it != modeOptionMap.end(); ++map_it) {
        map_it1 = map_it;
        ++map_it;
        if (map_it == modeOptionMap.end())
            break;
        for (map_it2 = map_it; map_it2 != modeOptionMap.end(); ++map_it2) {
            const string mode1 = canonize_name(map_it1->first);
            const string mode2 = canonize_name(map_it2->first);

            check_modes_gen.set_mode1_name(mode1);
            check_modes_gen.set_mode2_name(mode2);

            ostringstream mode1_given, mode2_given, mode1_options, mode2_options;

            std::for_each(map_it1->second.begin(), map_it1->second.end(), pair_print_f<OptionValueElem>(mode1_given, mode1_options));
            std::for_each(map_it2->second.begin(), map_it2->second.end(), pair_print_f<OptionValueElem>(mode2_given, mode2_options));

            check_modes_gen.set_mode1_given_fields(mode1_given.str());
            check_modes_gen.set_mode1_options(mode1_options.str());
            check_modes_gen.set_mode2_given_fields(mode2_given.str());
            check_modes_gen.set_mode2_options(mode2_options.str());

            check_modes_gen.generate_check_modes(stream, indent);
        }
        map_it = map_it1;
    }
}

void
CmdlineParserCreator::generate_clear_arg(ostream &stream, unsigned int indent)
{
  struct gengetopt_option * opt;
  clear_arg_gen_class clear_arg;

  /* now we initialize value fields */
  foropt
    {
      if (opt->type == ARG_NO)
        continue;

      clear_arg.set_name(opt->var_arg);
      clear_arg.set_suffix("arg");
      clear_arg.set_value("NULL");
      clear_arg.set_has_orig(opt->type != ARG_FLAG);
      clear_arg.set_has_arg(false);

      if (opt->multiple && opt->type)
        {
          clear_arg.set_has_arg(true);
        }
      else if (opt->type == ARG_STRING)
        {
          clear_arg.set_has_arg(true);
          if (opt->default_given)
            clear_arg.set_value
                ("gengetopt_strdup (\"" + string(opt->default_string) +
                "\")");
        }
      else if (opt->type == ARG_FLAG)
        {
          clear_arg.set_has_arg(true);
          clear_arg.set_suffix("flag");
          clear_arg.set_value(opt->flagstat ? "1" : "0");
        }
      else if (opt->type == ARG_ENUM)
      {
        // initialize enum arguments to -1 (unless they have a default)
        clear_arg.set_has_arg(true);
        if (opt->default_given)
            clear_arg.set_value(from_value_to_enum(opt->var_arg, opt->default_string));
        else
            clear_arg.set_value(string(opt->var_arg) + "__NULL");
      }
      else if (opt->default_given)
        {
          clear_arg.set_has_arg(true);
          clear_arg.set_value(opt->default_string);
        }

      clear_arg.generate_clear_arg(stream, indent);
    }
}

void
CmdlineParserCreator::generate_long_option_struct(ostream &stream,
                                                  unsigned int indent)
{
  string indent_str (indent, ' ');
  struct gengetopt_option * opt;

  foropt
    {
      stream << indent_str;

      stream << "{ \"" << opt->long_opt << "\",\t"
             << (opt->type == ARG_NO || opt->type == ARG_FLAG ? 0 :
                 (opt->arg_is_optional ? 2 : 1))
             << ", NULL, ";

      if (opt->short_opt)
        stream << "\'" << opt->short_opt << "\'";
      else
        stream << "0";

      stream << " }," << endl;
    }
}

string
CmdlineParserCreator::generate_getopt_string()
{
  struct gengetopt_option * opt;
  ostringstream built_getopt_string;

  foropt
    if (opt->short_opt)
      {
        built_getopt_string << opt->short_opt <<
          (opt->type == ARG_NO || opt->type == ARG_FLAG ? "" : ":");
        built_getopt_string <<
          (opt->arg_is_optional ? ":" : "");
      }

  return built_getopt_string.str ();
}

void
CmdlineParserCreator::generate_handle_no_short_option(ostream &stream,
                                                      unsigned int indent)
{
  handle_options(stream, indent, false);
}

void
CmdlineParserCreator::generate_handle_option(ostream &stream,
                                             unsigned int indent)
{
  handle_options(stream, indent, true);
}

void
CmdlineParserCreator::handle_options(ostream &stream, unsigned int indent, bool has_short)
{
  struct gengetopt_option * opt;
  generic_option_gen_class option_gen;
  string indent_str (indent, ' ');
  bool first = true;

  option_gen.set_has_short_option (has_short);

  // by default we handle '?' case in the switch
  // unless the user defined a short option as ?
  set_handle_question_mark(true);

  foropt
    {
      if (opt->short_opt == '?')
          set_handle_question_mark(false);

      if ((has_short && opt->short_opt) || (!has_short && !opt->short_opt))
        {
          if (has_short || first)
            stream << indent_str;

          option_gen.set_option_comment (opt->desc);
          option_gen.set_long_option (opt->long_opt);
          option_gen.set_short_option(opt->short_opt ? string (1, opt->short_opt) : "-");
          option_gen.set_option_var_name (opt->var_arg);
          option_gen.set_final_instructions("");

          if (!no_help && ((opt->short_opt == HELP_SHORT_OPT &&
                  strcmp(opt->long_opt, HELP_LONG_OPT) == 0)
                  || strcmp(opt->long_opt, HELP_LONG_OPT) == 0
                  || strcmp(opt->long_opt, FULL_HELP_LONG_OPT) == 0
                  || strcmp(opt->long_opt, DETAILED_HELP_LONG_OPT) == 0)) {
              bool full_help = (strcmp(opt->long_opt, FULL_HELP_LONG_OPT) == 0);
              bool detailed_help = (strcmp(opt->long_opt, DETAILED_HELP_LONG_OPT) == 0);
              if (no_handle_help) {
                    // we use the final_instructions parameter to call the free function
                    // and to return 0
                    const string final_instructions =
                    parser_function_name +
                    string("_free (&local_args_info);\nreturn 0;");

                    option_gen.set_final_instructions(final_instructions);

                    if (full_help) {
                        option_gen.set_long_option (FULL_HELP_LONG_OPT);
                        option_gen.set_option_comment (FULL_HELP_OPT_DESCR);
                    } else if (detailed_help) {
                        option_gen.set_long_option (DETAILED_HELP_LONG_OPT);
                        option_gen.set_option_comment (DETAILED_HELP_OPT_DESCR);
                    } else {
                        option_gen.set_long_option (HELP_LONG_OPT);
                        option_gen.set_short_option (HELP_SHORT_OPT_STR);
                        option_gen.set_option_comment (HELP_OPT_DESCR);
                    }
                    //option_gen.set_has_short_option (!full_help);
              } else {
                  handle_help_gen_class help_gen;
                  help_gen.set_parser_name (parser_function_name);
                  help_gen.set_full_help(full_help);
                  help_gen.set_detailed_help(detailed_help);
                  help_gen.set_short_opt(opt->short_opt == HELP_SHORT_OPT);
                  help_gen.generate_handle_help (stream, indent);
                  stream << endl;
                  stream << endl;
                  continue;
              }
          }

          if (!no_version && ((opt->short_opt == VERSION_SHORT_OPT && strcmp(opt->long_opt, VERSION_LONG_OPT) == 0)
                  || strcmp(opt->long_opt, VERSION_LONG_OPT) == 0)) {
              if (no_handle_version) {
                  option_gen.set_long_option (VERSION_LONG_OPT);
                  option_gen.set_short_option (VERSION_SHORT_OPT_STR);
                  option_gen.set_option_comment (VERSION_OPT_DESCR);
                  //option_gen.set_has_short_option (true);

                  // we use the final_instrauctions parameter to call the free function
                  // and to return 0
                  const string final_instructions =
                      parser_function_name +
                      string("_free (&local_args_info);\nreturn 0;");

                  option_gen.set_final_instructions(final_instructions);
              } else {
                  handle_version_gen_class version_gen;
                  version_gen.set_parser_name (parser_function_name);
                  version_gen.set_short_opt (opt->short_opt == VERSION_SHORT_OPT);
                  version_gen.generate_handle_version (stream, indent);
                  stream << endl;
                  stream << endl;
                  continue;
              }
          }

          if (opt->acceptedvalues != 0)
              option_gen.set_possible_values (OPTION_VALUES_NAME(opt->var_arg));
          else
              option_gen.set_possible_values ("0");

          string default_string = "0";
          if (opt->default_string)
              default_string = string("\"") + opt->default_string + "\"";
          option_gen.set_default_value (default_string);

          option_gen.set_arg_type(arg_type_constants[opt->type]);

          if (opt->group_value) {
              option_gen.set_group_var_name (canonize_name (opt->group_value));
              option_gen.set_option_has_group(true);
          } else
              option_gen.set_option_has_group(false);

          if (opt->mode_value) {
              // we reuse the variable group_var_name also for modes
              option_gen.set_group_var_name (canonize_name (opt->mode_value));
              option_gen.set_option_has_mode(true);
          } else
              option_gen.set_option_has_mode(false);

          option_gen.set_option_has_type(opt->type != 0);

          if (opt->multiple) {
              option_gen.set_multiple(true);
              option_gen.set_structure (string (opt->var_arg) + "_list");
          } else {
              option_gen.set_multiple(false);
              option_gen.set_structure (ARGS_STRUCT);
          }

          option_gen.generate_generic_option (stream, indent);

          if (has_short)
            {
              stream << endl;
            }

          if (first && !has_short)
            {
              first = false;
              option_gen.set_gen_else ("else ");
            }
        }
    }

  if (! first && !has_short) // something has been generated
    {
      generateBreak(stream, indent);
      stream << endl;
    }
}

#define GROUP_REQUIRED_COMPARISON "!="
#define GROUP_NOT_REQUIRED_COMPARISON ">"
#define GROUP_REQUIRED_MESSAGE "One"
#define GROUP_NOT_REQUIRED_MESSAGE "At most one"

void
CmdlineParserCreator::generate_handle_group(ostream &stream,
                                            unsigned int indent)
{
  group_option_gen_class opt_gen;
  string indent_str (indent, ' ');
  opt_gen.set_package_var_name (EXE_NAME);

  opt_gen.set_Comparison_rule(GROUP_NOT_REQUIRED_COMPARISON " 1");

  groups_collection_t::const_iterator end = gengetopt_groups.end();
  for ( groups_collection_t::const_iterator idx = gengetopt_groups.begin();
        idx != end; ++idx)
    {
      stream << indent_str;
      opt_gen.set_group_name (idx->first);
      opt_gen.set_group_var_name (canonize_name (idx->first));
      if (idx->second.required)
        {
          opt_gen.set_number_required(GROUP_REQUIRED_MESSAGE);
        }
      else
        {
          opt_gen.set_number_required(GROUP_NOT_REQUIRED_MESSAGE);
        }

      opt_gen.generate_group_option (stream, indent);
      stream << endl;
    }
}

void
CmdlineParserCreator::generate_handle_required(ostream &stream,
                                               unsigned int indent)
{
  struct gengetopt_option * opt;
  required_option_gen_class opt_gen;
  opt_gen.set_package_var_name ("prog_name");

  /* write test for required options or for multiple options
     (occurrence number check) */
  foropt
    if ( opt->required || opt->multiple )
      {
        if (opt->mode_value) {
            opt_gen.set_mode_condition("args_info->" +
                    canonize_name(opt->mode_value) + "_mode_counter && ");
        } else {
            opt_gen.set_mode_condition("");
        }

        // build the option command line representation
        ostringstream req_opt;
        req_opt << "'--" << opt->long_opt << "'";
        if (opt->short_opt)
          req_opt << " ('-" << opt->short_opt << "')";

        opt_gen.set_option_var_name (opt->var_arg);
        opt_gen.set_option_descr (req_opt.str ());

        // if the option is required this is the standard check
        if (opt->required) {
          opt_gen.set_checkrange(false);

          opt_gen.generate_required_option (stream, indent);
        }

        // if the option is multiple we generate also the
        // occurrence range check
        if (opt->multiple) {
          opt_gen.set_checkrange(true);

          opt_gen.generate_required_option (stream, indent);
        }

        // notice that the above ifs are not mutual exclusive:
        // a multiple option can have a range check without being
        // required.
      }

  // now generate the checks for required group options
  group_option_gen_class group_opt_gen;
  group_opt_gen.set_package_var_name ("prog_name");

  group_opt_gen.set_Comparison_rule("== 0");
  group_opt_gen.set_number_required(GROUP_REQUIRED_MESSAGE);

  groups_collection_t::const_iterator end = gengetopt_groups.end();
  for ( groups_collection_t::const_iterator idx = gengetopt_groups.begin();
        idx != end; ++idx)
  {
    if (idx->second.required)
    {
      group_opt_gen.set_group_name (idx->first);
      group_opt_gen.set_group_var_name (canonize_name (idx->first));

      group_opt_gen.generate_group_option (stream, indent);
      stream << endl;
    }
  }
}

void
CmdlineParserCreator::generate_handle_dependencies(ostream &stream,
                                               unsigned int indent)
{
  struct gengetopt_option * opt;
  dependant_option_gen_class opt_gen;
  opt_gen.set_package_var_name ("prog_name");
  string indent_str (indent, ' ');

  /* write test for required options */
  foropt
    if ( opt->dependon )
      {
        stream << indent_str;

        ostringstream req_opt;
        req_opt << "'--" << opt->long_opt << "'";
        if (opt->short_opt)
          req_opt << " ('-" << opt->short_opt << "')";

        opt_gen.set_option_var_name (opt->var_arg);
        opt_gen.set_dep_option (canonize_name(opt->dependon));
        opt_gen.set_option_descr (req_opt.str ());
        opt_gen.set_dep_option_descr (opt->dependon);

        opt_gen.generate_dependant_option (stream, indent);

        stream << endl;
      }
}

template <typename Collection>
void generate_counters(const Collection &collection, const string &name, ostream &stream, unsigned int indent)
{
    group_counter_gen_class counter_gen;
    string indent_str (indent, ' ');

    counter_gen.set_name(name);

    typename Collection::const_iterator end = collection.end();
    for ( typename Collection::const_iterator idx = collection.begin(); idx != end; ++idx) {
        stream << indent_str;
        counter_gen.set_group_name (canonize_name (idx->first));
        counter_gen.generate_group_counter (stream, indent);
        stream << endl;
    }
}

void
CmdlineParserCreator::generate_group_counters(ostream &stream,
                                              unsigned int indent)
{
    generate_counters(gengetopt_groups, "group", stream, indent);
}

void
CmdlineParserCreator::generate_mode_counters(ostream &stream,
                                              unsigned int indent)
{
    // we can reuse group counter gen class also for modes
    generate_counters(gengetopt_modes, "mode", stream, indent);
}

int
CmdlineParserCreator::generate_source ()
{
  /* ****************************************************** */
  /* ********************************************** C FILE  */
  /* ****************************************************** */

  set_usage_string (generate_usage_string ());
  set_getopt_string (generate_getopt_string ());

  string output_source = c_filename;

  if (src_output_dir.size())
      output_source = src_output_dir + "/" + output_source;
  else if (output_dir.size())
      output_source = output_dir + "/" + output_source;

  ofstream *output_file = open_fstream (output_source.c_str());
  generate_c_source (*output_file);
  output_file->close ();
  delete output_file;

  return 0;
}

void
CmdlineParserCreator::generate_free(ostream &stream,
                                    unsigned int indent)
{
  struct gengetopt_option * opt;

  foropt
    {
      free_option (opt, stream, indent);
    }
}

void
CmdlineParserCreator::generate_list_free(ostream &stream,
                                         unsigned int indent)
{
  struct gengetopt_option * opt;

  if (! has_multiple_options())
    return;

  free_list_gen_class free_list;

  foropt
    {
      if (opt->multiple && opt->type) {
        free_list.set_list_name(opt->var_arg);
        free_list.set_string_list(opt->type == ARG_STRING);
        free_list.generate_free_list(stream, indent);
      }
    }
}

void
CmdlineParserCreator::generate_file_save_loop(ostream &stream, unsigned int indent)
{
  struct gengetopt_option * opt;

  file_save_multiple_gen_class file_save_multiple;
  file_save_gen_class file_save;

  const string suffix = "_orig";
  const string suffix_given = "_given";

  foropt {
    if (opt->multiple) {
      file_save_multiple.set_has_arg(opt->type != ARG_NO);
      file_save_multiple.set_opt_var(opt->var_arg);
      file_save_multiple.set_opt_name(opt->long_opt);
      file_save_multiple.set_values
          ((opt->acceptedvalues ? OPTION_VALUES_NAME(opt->var_arg) : "0"));

      file_save_multiple.generate_file_save_multiple(stream, indent);
    } else {
      file_save.set_opt_name(opt->long_opt);
      file_save.set_given(opt->var_arg + suffix_given);
      file_save.set_values
          ((opt->acceptedvalues ? OPTION_VALUES_NAME(opt->var_arg) : "0"));

      if (opt->type != ARG_NO && opt->type != ARG_FLAG) {
        file_save.set_arg(opt->var_arg + suffix + (opt->multiple ? " [i]" : ""));
      } else {
        file_save.set_arg("");
      }
      file_save.generate_file_save(stream, indent);
    }
  }
}


