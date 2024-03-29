/*
 * File automatically generated by
 * gengen 1.2 by Lorenzo Bettini
 * https://www.gnu.org/software/gengen
 */

#include "multiple_fill_array.h"

void
multiple_fill_array_gen_class::generate_multiple_fill_array(ostream &stream, unsigned int indent)
{
  string indent_str (indent, ' ');
  indent = 0;

  if (( default_value != "0" ))
    {
      stream << "multiple_default_value.";
      if (( arg_type == "ARG_STRING" ))
        {
          stream << "default_";
        }
      generate_string (type, stream, indent + indent_str.length ());
      stream << "_arg = ";
      generate_string (default_value, stream, indent + indent_str.length ());
      stream << ";";
      stream << "\n";
      stream << indent_str;
    }
  stream << "update_multiple_arg((void *)&(args_info->";
  generate_string (option_var_name, stream, indent + indent_str.length ());
  stream << "_arg),";
  stream << "\n";
  stream << indent_str;
  stream << "  &(args_info->";
  generate_string (option_var_name, stream, indent + indent_str.length ());
  stream << "_orig), args_info->";
  generate_string (option_var_name, stream, indent + indent_str.length ());
  stream << "_given,";
  stream << "\n";
  stream << indent_str;
  stream << "  local_args_info.";
  generate_string (option_var_name, stream, indent + indent_str.length ());
  stream << "_given, ";
  if (( default_value != "0" ))
    {
      stream << "&multiple_default_value";
    }
  else
    {
      stream << "0";
    }
  stream << ",";
  stream << "\n";
  stream << indent_str;
  indent = 2;
  stream << "  ";
  generate_string (arg_type, stream, indent + indent_str.length ());
  stream << ", ";
  generate_string (option_var_name, stream, indent + indent_str.length ());
  stream << "_list);";
}
