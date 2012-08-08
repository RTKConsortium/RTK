// deal with namespace problems

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif // HAVE_CONFIG_H

#ifdef HAVE_SSTREAM
#include <sstream>
#else
#include "includes/sstream"
#endif

#ifdef HAVE_NAMESPACES
using std::ostringstream;
#endif
