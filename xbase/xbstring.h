/* 

    Xbase project source code

    This file contains the Class definition for a xbString object.

    Copyright (C) 1997  Gary A. Kunkel   

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact:

      Mail:

        Technology Associates, Inc.
        XBase Project
        1455 Deming Way #11
        Sparks, NV 89434
        USA

      Email:

        xbase@techass.com
	xdb-devel@lists.sourceforge.net
	xdb-users@lists.sourceforge.net

      See our website at:

        xdb.sourceforge.net

*/

#ifndef __XBSTRING_H__
#define __XBSTRING_H__

#ifdef __GNUG__
#pragma interface
#endif

#ifdef __WIN32__
#include <xbase/xbconfigw32.h>
#else
#include <xbase/xbconfig.h>
#endif

#include <stdlib.h>
#include <iostream>

/*! \file xbstring.h
*/

//! xbString class
/*!
*/

class XBDLLEXPORT xbString {
public:
  enum {npos = -1};

  xbString();
  xbString(size_t size);
  xbString(char c);
  xbString(const char *s);
  xbString(const char *s, size_t maxlen);
  xbString(const xbString &s);

  virtual ~xbString();

  xbString &operator=(const xbString &s);
  xbString &operator=(const char *s);
  xbString &operator=(char c);
  
  bool isNull() const;
  bool isEmpty() const;
  size_t len() const;
  size_t length() const;

  void resize(size_t size);  
  xbString copy() const;
  
  xbString &sprintf(const char *format, ...);
  void setNum(long num);

  xbString& assign(const xbString& str, size_t pos = 0, int n = npos);
  xbString& assign(char* str, int n);
  char operator[](int n) { return data[n]; }
  char getCharacter( int n ) const { return data[n]; }
  operator const char *() const;
  xbString &operator+=(const char *s);
  xbString &operator+=(char c);
  xbString &operator-=(const char *s);
  void putAt(size_t pos, char c);

  const char *getData() const;
  const char *c_str() const;
  void toLowerCase();
  int pos(char c);
  int pos(const char* s);
  void trim();
  bool compare(char s);
  bool compare(const char *s);

  bool operator == ( const xbString& ) const;
  bool operator != ( const xbString& ) const;
  bool operator < ( const xbString& ) const;
  bool operator > ( const xbString& ) const;
  bool operator <= ( const xbString& ) const;
  bool operator >= ( const xbString& ) const;

  friend XBDLLEXPORT std::ostream& operator << ( std::ostream&,
                                                 const xbString& );

  xbString &remove(size_t pos = 0, int n = npos);
  xbString mid(size_t pos = 0, int n = npos) const;

protected:
  void ctor(const char *s);
  void ctor(const char *s, size_t maxlen);

  char *data;
  size_t size;
  static const char * NullString;
};

XBDLLEXPORT xbString operator-(const xbString &s1, const xbString &s2);
XBDLLEXPORT xbString operator+(const xbString &s1, const xbString &s2);
XBDLLEXPORT xbString operator+(const xbString &s1, const char *s2);
XBDLLEXPORT xbString operator+(const char *s1, const xbString &s2);
XBDLLEXPORT xbString operator+(const xbString &s1, char c2);
XBDLLEXPORT xbString operator+(char c1, const xbString &s2);
XBDLLEXPORT bool operator==(const xbString &s1, const char *s2);
XBDLLEXPORT bool operator!=(const xbString &s1, const char *s2);

#endif

