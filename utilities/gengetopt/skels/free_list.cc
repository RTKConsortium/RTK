/*
 * File automatically generated by
 * gengen 1.2 by Lorenzo Bettini
 * https://www.gnu.org/software/gengen
 */

#include "free_list.h"

void
free_list_gen_class::generate_free_list(ostream &stream, unsigned int indent)
{
  string indent_str (indent, ' ');
  indent = 0;

  stream << "free_list (";
  generate_string (list_name, stream, indent + indent_str.length ());
  stream << "_list, ";
  if (string_list)
    {
      stream << "1 ";
    }
  else
    {
      stream << "0 ";
    }
  stream << ");";
  stream << "\n";
  stream << indent_str;
}
