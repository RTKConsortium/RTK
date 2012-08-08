//
// C++ Implementation: gm_utils
//
// Description:
//
//
// Author: Lorenzo Bettini <http://www.lorenzobettini.it>, (C) 2004
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string.h>

#include "gm_utils.h"
#include "my_sstream.h"
#include "ggo_options.h"
#include "argsdef.h"
#include "groups.h"

extern groups_collection_t gengetopt_groups;

char *
canonize_names(const char *name) {
    char *pvar;
    char *p;

    pvar = strdup(name);

    for (p = pvar; *p; ++p)
        if (*p == '.' || *p == '-')
            *p = '_';

    return pvar;
}

// remove the path from the file name
const string strip_path(const string &s) {
    string::size_type pos_of_sep;

    pos_of_sep = s.rfind("/");
    if (pos_of_sep == string::npos)
        pos_of_sep = s.rfind("\\"); // try also with DOS path sep

    if (pos_of_sep == string::npos)
        return s; // no path

    return s.substr(pos_of_sep + 1);
}

const string to_upper(const string &old) {
    string upper = old;

    for (string::iterator s = upper.begin(); s != upper.end(); ++s)
        *s = toupper(*s);

    return upper;
}

const string canonize_name(const string &old) {
    string canonized = old;

    for (string::iterator s = canonized.begin(); s != canonized.end(); ++s)
        if (*s == '.' || *s == '-' || *s == ' ')
            *s = '_';

    return canonized;
}

const string canonize_enum(const string &old) {
    string canonized;

    for (string::const_iterator s = old.begin(); s != old.end(); ++s)
        if (*s == '-')
            canonized += "MINUS_";
        else if (*s == '+')
            canonized += "PLUS_";
        else
            canonized += *s;

    return canonized;
}

bool has_multiple_options_all_string() {
    if (!has_multiple_options())
        return false;

    struct gengetopt_option * opt = 0;

    foropt {
        if (opt->multiple && (opt->type && opt->type != ARG_STRING))
            return false;
    }

    return true;
}

bool has_multiple_options_string() {
    if (!has_multiple_options())
        return false;

    struct gengetopt_option * opt = 0;

    foropt {
        if (opt->multiple && opt->type == ARG_STRING)
            return true;
    }

    return false;
}

bool has_multiple_options() {
    struct gengetopt_option * opt = 0;

    foropt {
        if (opt->multiple)
            return true;
    }

    return false;
}

bool has_multiple_options_with_type() {
    gengetopt_option * opt = 0;

    for (gengetopt_option_list::iterator it = gengetopt_options.begin(); it
            != gengetopt_options.end() && (opt = *it); ++it)
        if (opt->multiple && opt->type)
            return true;

    return false;
}

bool has_multiple_options_with_default() {
    gengetopt_option * opt = 0;

    for (gengetopt_option_list::iterator it = gengetopt_options.begin(); it
            != gengetopt_options.end() && (opt = *it); ++it)
        if (opt->multiple && opt->default_given)
            return true;

    return false;
}

bool has_options_with_details() {
    gengetopt_option * opt = 0;

    for (gengetopt_option_list::iterator it = gengetopt_options.begin(); it
            != gengetopt_options.end() && (opt = *it); ++it)
        if (opt->details)
            return true;

    return false;
}

bool has_options_with_type() {
    gengetopt_option * opt = 0;

    for (gengetopt_option_list::iterator it = gengetopt_options.begin(); it
            != gengetopt_options.end() && (opt = *it); ++it)
        if (opt->type && opt->type != ARG_FLAG)
            return true;

    return false;
}

bool has_options_with_mode() {
    gengetopt_option * opt = 0;

    for (gengetopt_option_list::iterator it = gengetopt_options.begin(); it
            != gengetopt_options.end() && (opt = *it); ++it)
        if (opt->mode_value)
            return true;

    return false;
}

bool has_required() {
    struct gengetopt_option * opt = 0;

    foropt {
        if (opt->required)
            return true;
    }

    groups_collection_t::const_iterator end = gengetopt_groups.end();
    for (groups_collection_t::const_iterator idx = gengetopt_groups.begin(); idx
            != end; ++idx) {
        if (idx->second.required) {
            return true;
        }
    }

    return false;
}

bool has_dependencies() {
    struct gengetopt_option * opt = 0;

    foropt {
        if (opt->dependon)
            return true;
    }

    return false;
}

bool has_options() {
    struct gengetopt_option * opt = 0;

    foropt {
        if (opt->short_opt) {
            if (opt->short_opt != 'h' && opt->short_opt != 'V')
                return true;
        }
    }

    return false;
}

bool has_hidden_options() {
    struct gengetopt_option * opt = 0;

    foropt {
        if (opt->hidden) {
            return true;
        }
    }

    return false;
}

bool has_values() {
    struct gengetopt_option * opt = 0;

    for (gengetopt_option_list::iterator it = gengetopt_options.begin(); it
            != gengetopt_options.end(); ++it) {
        opt = *it;
        if (opt->acceptedvalues) {
            return true;
        }
    }

    return false;
}

int not_newlines(const string &buf, int &num_of_newlines) {
    num_of_newlines = 0;
    // searches for the first non newline char
    string::size_type notnewline = buf.find_first_not_of("\r\n");

    if (notnewline == string::npos) {
        // a string made only of newlines
        num_of_newlines = buf.size();
        return num_of_newlines;
    }

    if (notnewline) {
        // everything before the non newline char is a newline
        num_of_newlines = notnewline;
        return notnewline;
    }

    return 0;
}

void wrap_cstr(string& wrapped, unsigned int from_column,
        unsigned int second_indent, const string &orig) {
    int next_space = from_column;
    string next_word;
    const char * out_buf = orig.c_str();
    ostringstream stream;
    const unsigned int second_line_column = from_column + second_indent;
    string indent(second_line_column, ' ');
    int newline_chars = 0;
    int num_of_newlines = 0;

    while (*out_buf) {
        // check for a new line
        if (*out_buf) {
            if ((newline_chars = not_newlines(out_buf, num_of_newlines))) {
                for (int i = 1; i <= num_of_newlines; ++i)
                    stream << "\\n";

                out_buf += newline_chars;

                if (*out_buf) {
                    stream << indent;
                    next_space = second_line_column;
                    continue;
                } else
                    break;
            } else {
                stream << *out_buf++;
                next_space++;
            }
        }
        // search next whitespace, i.e., next word
        while ((*out_buf) && (*out_buf != ' ') &&
                !not_newlines(out_buf, num_of_newlines)) {
            next_word += *out_buf++;
            next_space++;
        }

        // wrap line if it's too long
        if (next_space > 79) {
            stream << "\\n" << indent << next_word;
            next_space = second_line_column + next_word.size();
        } else
            stream << next_word; // simply write word

        next_word = "";
    }

    wrapped += stream.str();
}

bool is_numeric(const gengetopt_option *opt) {
    switch (opt->type) {
    case ARG_INT:
    case ARG_SHORT:
    case ARG_LONG:
    case ARG_FLOAT:
    case ARG_DOUBLE:
    case ARG_LONGDOUBLE:
    case ARG_LONGLONG:
        return true;
    default:
        return false;
    }
}
