/*
 * File automatically generated by
 * gengen 1.4.1 by Lorenzo Bettini
 * https://www.gnu.org/software/gengen
 */

#include "clear_given.h"

void
clear_given_gen_class::generate_clear_given(ostream &stream, unsigned int indent)
{
  string indent_str (indent, ' ');
  indent = 0;

  generate_string (arg_struct, stream, indent + indent_str.length ());
  stream << "->";
  generate_string (var_arg, stream, indent + indent_str.length ());
  stream << "_given = 0 ;";
  if (group)
    {
      indent = 1;
      stream << " ";
      generate_string (arg_struct, stream, indent + indent_str.length ());
      stream << "->";
      generate_string (var_arg, stream, indent + indent_str.length ());
      stream << "_group = 0 ;";
    }
  stream << "\n";
  stream << indent_str;
}
