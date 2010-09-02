/*  $Id: xbexcept.h,v 1.7 2003/08/16 19:59:39 gkunkel Exp $

    Xbase project source code

    This file contains definitions for xbase exceptions.

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
        xdb-devel@lits.sourceforge.net
	xdb-users@lists.sourceforge.net

      See our website at:

        xdb.sourceforge.net
*/

#ifndef __XBEXCEPT_H__
#define __XBEXCEPT_H__

#ifdef __GNUG__
#pragma interface
#endif

#ifdef __WIN32__
#include <xbase/xbconfigw32.h>
#else
#include <xbase/xbconfig.h>
#endif

#include <xbase/xtypes.h>

/*! \file xbexcept.h
*/

const char *xbStrError(xbShort err);

#ifndef HAVE_EXCEPTIONS
#define xb_error(code) { return code;}
#define xb_io_error(code,name) { return code;}
#define xb_open_error(name) { return XB_OPEN_ERROR;}
#define xb_memory_error { return XB_NO_MEMORY;}
#define xb_eof_error { return XB_EOF;}
#else

#ifdef HAVE_EXCEPTION

#include <exception>
#elif HAVE_G___EXCEPTION_H
#include <g++/exception.h>
#elif
#error "Exceptions are unsupported on your system."
#endif

#ifdef __BORLANDC__
#define XB_THROW throw ()
using std::exception;
#else
#define XB_THROW
#endif

//! xbException class
/*!
*/

/* FIXME:   class exception is member of <stdexcept.h> -- willy */
class XBDLLEXPORT xbException : public exception {
public:
  xbException (int err);
  virtual ~xbException () XB_THROW;
  virtual const char* what() const XB_THROW;
  virtual const char *error();
  int getErr();
private:
  int err;
};

#define xb_error(code) {throw xbException(code);return (code);}

//! xbIOException class
/*!
*/

class XBDLLEXPORT xbIOException : public xbException {
public:
  xbIOException (int err);
  xbIOException (int err, const char *n);
  virtual ~xbIOException () XB_THROW;
  virtual const char* what() const XB_THROW;
  const char *_errno() const;
  const char *name;
protected:
  int m_errno;
};

#define xb_io_error(code, name) {throw xbIOException(code,name);return (code);}

//! xbOpenException class
/*!
*/

class XBDLLEXPORT xbOpenException : public xbIOException {
public:
  xbOpenException ();
  xbOpenException (const char *n);
  virtual ~xbOpenException () XB_THROW;
  virtual const char* what() const XB_THROW;
};

#define xb_open_error(name) { throw xbOpenException(name); return 0;}

//! xbOutOfMemoryException class
/*!
*/

class XBDLLEXPORT xbOutOfMemoryException : public xbException {
public:
  xbOutOfMemoryException ();
  virtual ~xbOutOfMemoryException () XB_THROW;
  virtual const char* what() const XB_THROW;
};

#define xb_memory_error {throw xbOutOfMemoryException();return 0;}

//! xbEofException class
/*!
*/

class XBDLLEXPORT xbEoFException : public xbIOException {
public:
  xbEoFException ();
  virtual ~xbEoFException () XB_THROW;
  virtual const char* what() const XB_THROW;
};

#define xb_eof_error {throw xbEoFException();return 0;}

#endif

#endif // __XBEXCEPT_H__
