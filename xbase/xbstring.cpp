/* 
    Xbase project source code

    This file contains the xbString object methods

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

#ifdef __GNUG__
  #pragma implementation "xbstring.h"
#endif

#ifdef __WIN32__
#include <xbase/xbconfigw32.h>
#else
#include <xbase/xbconfig.h>
#endif

#include <xbase/xbase.h>

#include <stdlib.h>
#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STRINGS_H
#include <strings.h>
#endif

#ifdef STDC_HEADERS
#include <stdarg.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#include <xbase/xbstring.h>
#include <xbase/xbexcept.h>

//#define free(x)

/*! \file xbstring.cpp
*/

const char * xbString::NullString = "";

//! Short description.
/*!
*/
xbString::xbString() {
  ctor(NULL);
}

//! Short description.
/*!
  \param size
*/
xbString::xbString(size_t size) {
  data = (char *)calloc(1, size);
  this->size = size;
}

//! Short description.
/*!
  \param c
*/
xbString::xbString(char c) {
  ctor(NULL);
  *this = c;
}

//! Short description.
/*!
  \param s
*/
xbString::xbString(const char *s) {
  ctor(s);
}

//! Short description.
/*!
  \param s
*/
xbString::xbString(const xbString &s) {
  ctor((const char *)s);
}

//! Short description.
/*!
  \param s
  \param maxlen
*/
xbString::xbString(const char *s, size_t maxlen) {
#if 0
  size_t len = strlen(s);
  
  if(len < maxlen)
    maxlen = len;
#endif    

  size = maxlen + 1;
  data = (char *)calloc(1, size);
  strncpy(data, s, maxlen);
  data[maxlen] = 0;
}

//! Short description.
/*!
*/
xbString::~xbString() {
  if (data != NULL)
    free(data);
}

//! Short description.
/*!
  \param s
*/
void xbString::ctor(const char *s) {
  if (s == NULL) {
    data = NULL;
    size = 0;
    return;
  }

  size = strlen(s) + 1;

  data = (char *)calloc(1, size);
  strcpy(data, s);
}

//! Short description.
/*!
  \param s
  \param maxlen
*/
void xbString::ctor(const char *s, size_t maxlen) {

  if (s == NULL) {
    data = NULL;
    size =0;
    return;
  }

#if 0
  size_t len = strlen(s);
  
  if (len < maxlen)
    maxlen = len;
#endif    

  size = maxlen + 1;
  data = (char *)calloc(1, size);
  strncpy(data, s, maxlen);
  data[maxlen] = 0;
}

//! Short description.
/*!
*/
xbString &xbString::operator=(char c) {
  if (data != NULL)
    free(data);

  data = (char *)calloc(1, 2);
  data[0] = c;
  data[1] = 0;

  size = 2;

  return (*this);
}

//! Short description.
/*!
*/
xbString &xbString::operator=(const xbString &s) {
  if (data != NULL)
    free(data);

  const char *sd = s;
  if (sd == NULL) {
    data = NULL;
    size = 0;
    return (*this);
  }

  data = (char *)calloc(1, strlen(s) + 1);
  strcpy(data, s);

  size = strlen(data)+1;
  
  return (*this);
}

//! Short description.
/*!
*/
xbString &xbString::operator=(const char *s) {
  if (data != NULL)
    free(data);

   if (s == NULL) {
      data = NULL;
      size = 0;
      return (*this);
   }

  data = (char *)calloc(1, strlen(s) + 1);
  strcpy(data, s);

  size = strlen(data) + 1;
  
  return (*this);
}

//! Short description.
/*!
  \param size
*/
void xbString::resize(size_t size) {
  data = (char *)realloc(data, size);
  if (size>0)
    data[size-1] = 0;
  this->size = size;
}
     
//! Short description.
/*!
*/
bool xbString::isNull() const {
  return (data == NULL);
}

//! Short description.
/*!
*/
bool xbString::isEmpty() const {
  if (data == NULL)
    return true;
  if (data[0] == 0)
    return true;
  return false;
}

//! Short description.
/*!
*/
size_t xbString::len() const {
  return (data ? strlen(data) : 0);
}

//! Short description.
/*!
*/
size_t xbString::length() const {
  return len();
}

//! Short description.
/*!
*/
xbString xbString::copy() const {
  return (*this);
}

//! Short description.
/*!
  \param format
*/
xbString &xbString::sprintf(const char *format, ...) {
  va_list ap;
  va_start(ap, format);

  if (size < 256)
    resize(256);              // make string big enough

#ifdef HAVE_VSNPRINTF
  if (vsnprintf(data, size, format, ap) == -1)
    data[size-1] = 0;
#else
#  if HAVE_VSPRINTF
  vsprintf(data, format, ap);
#  else
#    if
#      error "You have neither vsprintf nor vsnprintf!!!"
#    endif
#  endif
#endif

  resize(strlen(data)+1);               // truncate
  va_end(ap);
  return (*this);
}

//! Short description.
/*!
*/
xbString::operator const char *() const {
//  return data;
  return (data != NULL) ? data : NullString;
}

//! Short description.
/*!
*/
xbString &xbString::operator-=(const char *s) {
  if( s == NULL ) return (*this);
  int len = strlen(s);
  int oldlen = this->len();

  data = (char *)realloc(data, oldlen+len+1);
  if( oldlen == 0 ) data[0] = 0;

  // looking for an occurence of space in the first string
  char *lftspc = strchr(data,' ');
  if( lftspc==NULL ) { // left string has no spaces
    strcat(data,s);
  } else { // left string has one or more spaces
    int numspc = strlen(lftspc);
    strcpy(lftspc,s);
    while( numspc-- > 0 ) strcat(lftspc," ");
  }

  size += len;
  return (*this);
}

//! Short description.
/*!
*/
xbString &xbString::operator+=(const char *s) {
  if (s == NULL)
    return (*this);
  int len = strlen(s);
  int oldlen = this->len();

  data = (char *)realloc(data, oldlen+len+1);
   if (oldlen == 0)
      data[0] = 0;
  strcat(data, s);

  size += len;
  return (*this);
}

//! Short description.
/*!
*/
xbString &xbString::operator+=(char c) {
  int len = 1;
  int oldlen = this->len();

  data = (char *)realloc(data, oldlen+len+1);
  data[oldlen] = c;
  data[oldlen+1] = 0;

  size++;

  return (*this);
}

//! Short description.
/*!
*/
const char *xbString::getData() const {
  return data ? data : NullString;
}

//! Short description.
/*!
*/
const char *xbString::c_str() const {
  return data ? data : NullString;
}

//! Short description.
/*!
*/
void xbString::toLowerCase() {
  int len = this->len();
  for (int i=0;i<len;i++)
    data[i] = (char)tolower(data[i]);
}

//! Short description.
/*!
  \param c
*/
int xbString::pos(char c) {
  if (data == NULL)
    return (-1);

  const char *p = strchr(data, c);

  if (p == NULL)
    return (-1);

  return p-data;
}

//! Short description.
/*!
  \param s
*/
int xbString::pos(const char* s) {
  if (data == NULL)
    return (-1);

  const char *p = strstr(data, s);

  if (p == NULL)
    return (-1);

  return p-data;
}

//! Short description.
/*!
  \param num
*/
void xbString::setNum(long num) {
  sprintf("%ld", num);
}

//! Short description.
/*!
*/
XBDLLEXPORT bool operator==(const xbString &s1, const char *s2) {
  if (s2 == NULL) {
    if (s1.getData() == NULL)
      return true;
    return false;
  }

   if ((s2[0] == 0) && s1.getData() == NULL)
      return true;

  if (s1.getData() == NULL)
    return false;

  return (strcmp(s1, s2) == 0);
}

//! Short description.
/*!
*/
XBDLLEXPORT bool operator!=(const xbString &s1, const char *s2) {
  if (s2 == NULL) {
    if (s1.getData() == NULL)
      return false;
    return true;
  }

   if ((s2[0] == 0) && s1.getData() == NULL)
      return false;

  if (s1.getData() == NULL)
    return true;

  return (strcmp(s1, s2) != 0);
}

//! Short description.
/*!
*/
bool xbString::operator==( const xbString &s2 ) const {
  if( data == NULL || data[0] == 0 ) {
    if( s2.data == NULL || s2.data[0] == 0 ) return true; // NULL == NULL
    return false; // NULL == !NULL
  } else {
    if( s2.data == NULL || s2.data[0] == 0 ) return false; // !NULL == NULL
    return strcmp(data,s2.data) == 0; //!NULL == !NULL
  }
}

//! Short description.
/*!
*/
bool xbString::operator!=( const xbString &s2 ) const {
  if( data == NULL || data[0] == 0 ) {
    if( s2.data == NULL || s2.data[0] == 0 ) return false; // NULL != NULL
    return true; // NULL != !NULL
  } else {
    if( s2.data == NULL || s2.data[0] == 0 ) return true; // !NULL != NULL
    return strcmp(data,s2.data) != 0; //!NULL != !NULL
  }
}

//! Short description.
/*!
*/
bool xbString::operator< ( const xbString &s2 ) const {
  if( data == NULL || data[0] == 0 ) {
    if( s2.data == NULL || s2.data[0] == 0 ) return false; // NULL < NULL
    return true; // NULL < !NULL
  } else {
    if( s2.data == NULL || s2.data[0] == 0 ) return false; // !NULL < NULL
    return strcmp(data,s2.data) < 0; //!NULL < !NULL
  }
}

//! Short description.
/*!
*/
bool xbString::operator> ( const xbString &s2 ) const {
  if( data == NULL || data[0] == 0 ) {
    if( s2.data == NULL || s2.data[0] == 0 ) return false; // NULL > NULL
    return false; // NULL > !NULL
  } else {
    if( s2.data == NULL || s2.data[0] == 0 ) return true; // !NULL > NULL
    return strcmp(data,s2.data) > 0; //!NULL > !NULL
  }
}

//! Short description.
/*!
*/
bool xbString::operator<=( const xbString &s2 ) const {
  if( data == NULL || data[0] == 0 ) {
    if( s2.data == NULL || s2.data[0] == 0 ) return true; // NULL <= NULL
    return true; // NULL <= !NULL
  } else {
    if( s2.data == NULL || s2.data[0] == 0 ) return false; // !NULL <= NULL
    return strcmp(data,s2.data) <= 0; //!NULL <= !NULL
  }
}

//! Short description.
/*!
*/
bool xbString::operator>=( const xbString &s2 ) const {
  if( data == NULL || data[0] == 0 ) {
    if( s2.data == NULL || s2.data[0] == 0 ) return true; // NULL >= NULL
    return false; // NULL >= !NULL
  } else {
    if( s2.data == NULL || s2.data[0] == 0 ) return true; // !NULL >= NULL
    return strcmp(data,s2.data) >= 0; //!NULL >= !NULL
  }
}

//! Short description.
/*!
*/
XBDLLEXPORT std::ostream& operator<< ( std::ostream& os,
                                       const xbString& xbs ) {
  return os << xbs.data;
}

//! Short description.
/*!
*/
XBDLLEXPORT xbString operator-(const xbString &s1, const xbString &s2) {
   xbString tmp(s1.getData());
   tmp -= s2;
   return tmp;
}

//! Short description.
/*!
*/
XBDLLEXPORT xbString operator+(const xbString &s1, const xbString &s2) {
   xbString tmp(s1.getData());
   tmp += s2;
   return tmp;
}

//! Short description.
/*!
*/
XBDLLEXPORT xbString operator+(const xbString &s1, const char *s2) {
   xbString tmp(s1.getData());
   tmp += s2;
   return tmp;
}

//! Short description.
/*!
*/
XBDLLEXPORT xbString operator+(const char *s1, const xbString &s2) {
   xbString tmp(s1);
   tmp += s2;
   return tmp;
}

//! Short description.
/*!
*/
XBDLLEXPORT xbString operator+(const xbString &s1, char c2) {
   xbString tmp(s1.getData());
   tmp += c2;
   return tmp;
}

//! Short description.
/*!
*/
XBDLLEXPORT xbString operator+(char c1, const xbString &s2) {
   xbString tmp(c1);
   tmp += s2;
   return tmp;
}

//! Short description.
/*!
  \param pos
  \param c
*/
void xbString::putAt(size_t pos, char c) {
   if (pos>len())
      return;

   data[pos] = c;
}

//! Short description.
/*!
  \param str
  \param pos
  \param n
*/
xbString& xbString::assign(const xbString& str, size_t pos, int n)
{
  if(data)
  {
    free(data);
    data = 0;
  }

  if(str.len() <= pos)
  {
    size = 0;
    return (*this);
  }

  if(str.len() < pos + n)
  {
    n = str.len() - pos;
  }

  const char *d = str;

  if (n == -1)
  {
//        data = (char *)malloc(str.len()-pos+1); ms win/nt bug fix
    data = (char *)calloc(str.len()-pos+1, sizeof( char ));
    strcpy(data, d+pos);
    size = str.len()-pos+1;
  }
  else
  {
//   data = (char *)malloc(n);  ms win/nt bug fix
    data = (char *)calloc(n + 1, sizeof(char));
    strncpy(data, d + pos, n);
    data[n] = '\0';
    size = n + 1;
  }

  return (*this);
}

//! Short description.
/*!
  \param str
  \param pos
  \param n
*/
xbString& xbString::assign(char* str, int n)
{
  if(data)
  {
    free(data);
    data = 0;
  }

  data = (char *)calloc(n + 1, sizeof(char));
  strncpy(data, str, n);
  data[n] = 0;
  size = n + 1;

  return (*this);
}

//! Short description.
/*!
*/
void xbString::trim() {
  int l = len()-1;

   for (;;) {
      if (data[l] != ' ')
         break;
      data[l] = 0;
      if (l == 0)
         break;
      l--;
   }
}

//! Short description.
/*!
  \param pos
  \param n
*/
xbString &xbString::remove(size_t pos, int n) {
  if (data == NULL)
    return (*this);
  if (data[0] == 0)
    return (*this);

  size_t l = len();

  if (pos>l)
    return (*this);
  if (n == 0)
    return (*this);
  if (n > int(l-pos))
    n = l-pos;
  if (n<0)
    n = l-pos;
  memcpy(data+pos, data+pos+n, l-pos-n+1);

  return (*this);
}

//! Short description.
/*!
  \param pos
  \param n
*/
xbString xbString::mid(size_t pos, int n) const {
  if (data == NULL)
    return (*this);
  if (data[0] == 0)
    return (*this);

  size_t l = len();

  if (pos>l)
    return (*this);
  if (n == 0)
    return (*this);
  if (n > int(l-pos))
    n = l-pos;
  if (n<0)
    n = l-pos;

  xbString s;
  s.data = (char *)malloc(n+1);
  strncpy(s.data, data+pos, n);
  s.data[n] = 0;

  return s;
}

