#ifndef GGO_OPTIONS_H
#define GGO_OPTIONS_H

#include "ggos.h"

extern gengetopt_option_list gengetopt_options;

#define foropt for (gengetopt_option_list::iterator it = gengetopt_options.begin();             \
                    it != gengetopt_options.end() && (opt = *it); \
                    ++it)

#endif /* GGO_OPTIONS_H */
