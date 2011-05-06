//
// C++ Interface: acceptedvalues
//
// Description:
//
//
// Author: Lorenzo Bettini <http://www.lorenzobettini.it>, (C) 2004
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef ACCEPTEDVALUES_H
#define ACCEPTEDVALUES_H

#include <list>
#include <set>
#include <string>

/**
the values that can be passed to an option

@author Lorenzo Bettini
*/
class AcceptedValues : protected std::list<std::string>
{
  private:
    typedef std::set<std::string> value_set;
    value_set values;
  
  public:
    using std::list<std::string>::const_iterator;
    using std::list<std::string>::begin;
    using std::list<std::string>::end;

    void insert(const std::string &s);
    const std::string toString(bool escape = true) const;
    bool contains(const std::string &s) const;
};

#endif
