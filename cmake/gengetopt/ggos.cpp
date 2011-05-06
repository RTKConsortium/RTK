//
// C++ Implementation: ggos
//
// Description:
//
//
// Author: Lorenzo Bettini <http://www.lorenzobettini.it>, (C) 2005-2007
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include "ggos.h"

using namespace std;

gengetopt_option::gengetopt_option() :
    short_opt(0), long_opt(0), desc(0), type(ARG_NO), flagstat(-1),
            required(1), required_set(false), var_arg(0), default_string(0),
            group_value(0), group_desc(0),
            mode_value(0), mode_desc(0),
            multiple(false),
            arg_is_optional(false), hidden(false), type_str(0),
            acceptedvalues(0), section(0), section_desc(0), dependon(0),
            text_before(0), text_after(0), details(0), filename(0), linenum(0) {
}

ostream & operator <<(std::ostream &s, gengetopt_option &opt) {
    s << "long: " << opt.long_opt << ", short: " << opt.short_opt << "\n"
            << "desc: " << opt.desc;

    s << endl;

    return s;
}
