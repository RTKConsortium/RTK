/*  $Id: xbfilter.cpp

    Xbase project source code
  
    This file conatains a header file for the xbStack object which
    is used for handling expressions.

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
  #pragma implementation "xbfilter.h"
#endif

#ifdef __WIN32__
#include <xbase/xbconfigw32.h>
#else
#include <xbase/xbconfig.h>
#endif

#include <xbase/xbase.h>
#include <xbase/xbexcept.h>

/*! \file xbfilter.cpp
*/

#ifdef XB_FILTERS
/************************************************************************/
//! Constructor.
/*!
  \param dbf
  \param index
  \param exp
*/
xbFilter::xbFilter( xbDbf * dbf, xbIndex * index, char * exp )
{
  xbShort rc;
  Status = 0;
  CurFilterRecNo = 0L;
  d = dbf;
  i = index;
  e = 0;

  if(( rc = d->xbase->ParseExpression( exp, d )) != XB_NO_ERROR )
    Status = rc;
  else{
    e = d->xbase->GetExpressionHandle();
    if( d->xbase->GetExpressionResultType( e ) != 'L' )
      Status = XB_PARSE_ERROR;
  }
}

/***********************************************************************/
//! Destructor.
/*!
*/
xbFilter::~xbFilter()
{
  delete e;
}

/***********************************************************************/
//! Short description.
/*!
*/
xbShort xbFilter::GetFirstFilterRec()
{
  xbShort rc;

  if( Status )
    return Status;

  if( i )
    rc = i->GetFirstKey();
  else
    rc = d->GetFirstRecord();

  while( rc == XB_NO_ERROR ){
    if(( rc = d->xbase->ProcessExpression( e )) != XB_NO_ERROR )
      xb_error( rc );

    if(d->xbase->GetIntResult())
    {
      CurFilterRecNo = d->GetCurRecNo();
      return XB_NO_ERROR;
    }
    if( i )
      rc = i->GetNextKey();
    else
      rc = d->GetNextRecord();
  }
  return rc;
}
/***********************************************************************/
//! Short description.
/*!
*/
xbShort xbFilter::GetLastFilterRec()
{
  xbShort rc;

  if( Status )
    return Status;

  if( i )
    rc = i->GetLastKey();
  else
    rc = d->GetLastRecord();

  while( rc == XB_NO_ERROR ){
    if(( rc = d->xbase->ProcessExpression( e )) != XB_NO_ERROR )
      xb_error( rc );

    if(d->xbase->GetIntResult())
    {
      CurFilterRecNo = d->GetCurRecNo();
      return XB_NO_ERROR;
    }
    if( i )
      rc = i->GetPrevKey();
    else
      rc = d->GetPrevRecord();
  }
  return rc;
}
/***********************************************************************/
//! Short description.
/*!
*/
xbShort xbFilter::GetNextFilterRec()
{
  xbShort rc;

  if( Status )
    return Status;

  if( !CurFilterRecNo )
    return GetFirstFilterRec();

  if( i ){
    rc = i->GetNextKey();
  }
  else
    rc = d->GetNextRecord();

  while( rc == XB_NO_ERROR ){
    if(( rc = d->xbase->ProcessExpression( e )) != XB_NO_ERROR )
      xb_error( rc );

    if(d->xbase->GetIntResult())
    {
      CurFilterRecNo = d->GetCurRecNo();
      return XB_NO_ERROR;
    }
    if( i )
      rc = i->GetNextKey();
    else
      rc = d->GetNextRecord();
  }
  return rc;
}
/***********************************************************************/
//! Short description.
/*!
*/
xbShort xbFilter::GetPrevFilterRec()
{
  xbShort rc;

  if( Status )
    return Status;

  if( !CurFilterRecNo )
    return GetLastFilterRec();

  if( i ){
    rc = i->GetPrevKey();
  }
  else
    rc = d->GetPrevRecord();

  while( rc == XB_NO_ERROR ){
    if(( rc = d->xbase->ProcessExpression( e )) != XB_NO_ERROR )
      xb_error( rc );

    if(d->xbase->GetIntResult())
    {
      CurFilterRecNo = d->GetCurRecNo();
      return XB_NO_ERROR;
    }
    if( i )
      rc = i->GetPrevKey();
    else
      rc = d->GetPrevRecord();
  }
  return rc;
}
/***********************************************************************/
#endif  // XB_FILTERS_ON
