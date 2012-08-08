//
// C++ Implementation: acceptedvalues
//
// Description:
//
//
// Author: Lorenzo Bettini <http://www.lorenzobettini.it>, (C) 2004
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include "acceptedvalues.h"
#include "my_sstream.h"

using namespace std;

void
AcceptedValues::insert(const string &s)
{
  push_back(s);
  values.insert(s);
}

bool
AcceptedValues::contains(const string &s) const
{
  return (values.count(s) > 0);
}

const string
AcceptedValues::toString(bool escape) const
{
  ostringstream buf;

  for (const_iterator it = begin(); it != end(); ) {
    buf << (escape ? "\\\"" : "\"") << *it
        << (escape ? "\\\"" : "\"");
    if (++it != end())
      buf << ", ";
  }

  return buf.str();
}
