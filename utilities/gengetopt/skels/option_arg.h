/*
 * File automatically generated by
 * gengen 1.3 by Lorenzo Bettini
 * https://www.gnu.org/software/gengen
 */

#ifndef OPTION_ARG_GEN_CLASS_H
#define OPTION_ARG_GEN_CLASS_H

#include <string>
#include <iostream>

using std::string;
using std::ostream;

class option_arg_gen_class
{
 protected:
  bool default_on;
  string default_value;
  string desc;
  bool flag_arg;
  bool has_arg;
  bool has_default;
  bool has_enum;
  bool long_long_arg;
  string longtype;
  bool multiple;
  string name;
  string origtype;
  string type;

 public:
  option_arg_gen_class() :
    default_on (false), flag_arg (false), has_arg (false), has_default (false), has_enum (false), long_long_arg (false), multiple (false)
  {
  }

  option_arg_gen_class(bool _default_on, const string &_default_value, const string &_desc, bool _flag_arg, bool _has_arg, bool _has_default, bool _has_enum, bool _long_long_arg, const string &_longtype, bool _multiple, const string &_name, const string &_origtype, const string &_type) :
    default_on (_default_on), default_value (_default_value), desc (_desc), flag_arg (_flag_arg), has_arg (_has_arg), has_default (_has_default), has_enum (_has_enum), long_long_arg (_long_long_arg), longtype (_longtype), multiple (_multiple), name (_name), origtype (_origtype), type (_type)
  {
  }

  static void
  generate_string(const string &s, ostream &stream, unsigned int indent)
  {
    if (!indent || s.find('\n') == string::npos)
      {
        stream << s;
        return;
      }

    string::size_type pos;
    string::size_type start = 0;
    string ind (indent, ' ');
    while ( (pos=s.find('\n', start)) != string::npos)
      {
        stream << s.substr (start, (pos+1)-start);
        start = pos+1;
        if (start+1 <= s.size ())
          stream << ind;
      }
    if (start+1 <= s.size ())
      stream << s.substr (start);
  }

  void set_default_on(bool _default_on)
  {
    default_on = _default_on;
  }

  void set_default_value(const string &_default_value)
  {
    default_value = _default_value;
  }

  void set_desc(const string &_desc)
  {
    desc = _desc;
  }

  void set_flag_arg(bool _flag_arg)
  {
    flag_arg = _flag_arg;
  }

  void set_has_arg(bool _has_arg)
  {
    has_arg = _has_arg;
  }

  void set_has_default(bool _has_default)
  {
    has_default = _has_default;
  }

  void set_has_enum(bool _has_enum)
  {
    has_enum = _has_enum;
  }

  void set_long_long_arg(bool _long_long_arg)
  {
    long_long_arg = _long_long_arg;
  }

  void set_longtype(const string &_longtype)
  {
    longtype = _longtype;
  }

  void set_multiple(bool _multiple)
  {
    multiple = _multiple;
  }

  void set_name(const string &_name)
  {
    name = _name;
  }

  void set_origtype(const string &_origtype)
  {
    origtype = _origtype;
  }

  void set_type(const string &_type)
  {
    type = _type;
  }

  void generate_option_arg(ostream &stream, unsigned int indent = 0);

};

#endif // OPTION_ARG_GEN_CLASS_H
