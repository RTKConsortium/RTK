/*
 * File automatically generated by
 * gengen 1.2 by Lorenzo Bettini
 * https://www.gnu.org/software/gengen
 */

#include "handle_help.h"

void
handle_help_gen_class::generate_handle_help(ostream &stream, unsigned int indent)
{
  string indent_str (indent, ' ');
  indent = 0;

  if (detailed_help)
    {
      stream << "if (strcmp (long_options[option_index].name, \"detailed-help\") == 0) {";
      stream << "\n";
      stream << indent_str;
      indent = 2;
      stream << "  ";
      generate_string (parser_name, stream, indent + indent_str.length ());
      stream << "_print_detailed_help ();";
      indent = 0;
      stream << "\n";
      stream << indent_str;
    }
  else
    {
      if (full_help)
        {
          stream << "if (strcmp (long_options[option_index].name, \"full-help\") == 0) {";
          stream << "\n";
          stream << indent_str;
          indent = 2;
          stream << "  ";
          generate_string (parser_name, stream, indent + indent_str.length ());
          stream << "_print_full_help ();";
          indent = 0;
          stream << "\n";
          stream << indent_str;
        }
      else
        {
          if (short_opt)
            {
              stream << "case 'h':	/* Print help and exit.  */";
              stream << "\n";
              stream << indent_str;
            }
          else
            {
              stream << "if (strcmp (long_options[option_index].name, \"help\") == 0) {";
              stream << "\n";
              stream << indent_str;
            }
          stream << "  ";
          generate_string (parser_name, stream, indent + indent_str.length ());
          stream << "_print_help ();";
          stream << "\n";
          stream << indent_str;
        }
    }
  stream << "  ";
  generate_string (parser_name, stream, indent + indent_str.length ());
  stream << "_free (&local_args_info);";
  stream << "\n";
  stream << indent_str;
  stream << "  exit (EXIT_SUCCESS);";
  if (( full_help || ( ! short_opt ) ))
    {
      stream << "\n";
      stream << indent_str;
      stream << "}";
    }
}
