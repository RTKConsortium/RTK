/*
 * File automatically generated by
 * gengen 1.2 by Lorenzo Bettini
 * https://www.gnu.org/software/gengen
 */

#include "update_given.h"

void
update_given_gen_class::generate_update_given(ostream &stream, unsigned int indent)
{
  string indent_str (indent, ' ');
  indent = 0;

  stream << "args_info->";
  generate_string (option_var_name, stream, indent + indent_str.length ());
  stream << "_given += local_args_info.";
  generate_string (option_var_name, stream, indent + indent_str.length ());
  stream << "_given;";
  stream << "\n";
  stream << indent_str;
  stream << "local_args_info.";
  generate_string (option_var_name, stream, indent + indent_str.length ());
  stream << "_given = 0;";
  stream << "\n";
  stream << indent_str;
}
