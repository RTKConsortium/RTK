//
// C++ Interface: fileutils
//
// Description:
//
//
// Author: Lorenzo Bettini <http://www.lorenzobettini.it>, (C) 2004
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef FILEUTILS_H
#define FILEUTILS_H

#include <fstream>

using std::ofstream;

char *create_filename (char *name, char *ext);
ofstream *open_fstream (const char *filename);

#endif
