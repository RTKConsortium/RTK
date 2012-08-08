//
// C++ Interface: gm_utils
//
// Description:
//
//
// Author: Lorenzo Bettini <http://www.lorenzobettini.it>, (C) 2004
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef GM_UTILS_H
#define GM_UTILS_H

#include <string>
#include <iostream>

#include "ggos.h"

using std::string;

/**
 * @param name
 * @return a copy of the string passed after canonizing it (i.e. '-' and
 * '.' are transformed in '_').
 */
char *canonize_names(const char * name);

/**
 * @param name
 * @return a copy of the string passed after canonizing it (i.e. '-' and
 * '.' are transformed in '_').
 */
const string canonize_name(const string &name);

/**
 * @param s the string representing an enum value
 * @return a copy of the string passed after canonizing it (i.e. '-' and
 * becomes _MINUS_, '+' becomes _PLUS_)
 */
const string canonize_enum(const string &s);

const string strip_path(const string &);
const string to_upper(const string &);

/**
 * All multiple options are of type string
 * @return All multiple options are of type string
 */
bool has_multiple_options_all_string();

/**
 * Has multiple options and at least one is of type string
 * @return Has multiple options and at least one is of type string
 */
bool has_multiple_options_string();

/**
 * Has multiple options and at least one has a default value
 * @return Has multiple options and at least one has a default value
 */
bool has_multiple_options_with_default();

bool has_multiple_options();
bool has_multiple_options_with_type();
bool has_required();
bool has_dependencies();
bool has_options_with_type();
bool has_options_with_mode();
bool has_options();
bool has_hidden_options();
bool has_options_with_details();
bool has_values();

/**
 * Whether the specified option deals with number
 *
 * @param opt
 * @return
 */
bool is_numeric(const gengetopt_option *opt);

/**
 * Performs word wrapping on the passed string (and return the result in the first
 * parameter).
 *
 * @param wrapped the output parameter
 * @param from_column the string start from this column
 * @param second_indent an additional indentation for lines after the
 * first one
 * @param orig the original string that must be wrapped
 */
void wrap_cstr (string &wrapped, unsigned int from_column, unsigned int second_indent, const string &orig);

/**
 * Searches for characters which are not newlines.
 *
 * @param buf where to search for new characters
 * @param num_of_newlines where the number of newlines
 * before the first non newline char will be stored
 * @return the position in the string after the (possible) new line char
 */
int not_newlines(const string &buf, int &num_of_newlines);

/**
 * Function object to print something into a stream (to be used with for_each)
 */
template<class T>
struct print_f : public std::unary_function<T, void>
{
    print_f(std::ostream& out, const string &s = ", ") : os(out), sep(s) {}
    void operator() (T x) { os << x << sep; }
    std::ostream& os;
    const string &sep;
};

/**
 * Function object to print a pair into two streams (to be used with for_each)
 */
template<class T>
struct pair_print_f : public std::unary_function<T, void>
{
    pair_print_f(std::ostream& out1, std::ostream& out2, const string &s = ", ") :
        os1(out1), os2(out2), sep(s) {}
    void operator() (T x) { os1 << x.first << sep; os2 << x.second << sep;}
    std::ostream &os1, &os2;
    const string &sep;
};

#endif
