/*  $Id: index.cpp,v 1.7 2003/08/16 19:59:39 gkunkel Exp $

    Xbase project source code
   
    This file contains the implementation of the xbIndex class.

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
  #pragma implementation "index.h"
#endif

#ifdef __WIN32__
#include <xbase/xbconfigw32.h>
#else
#include <xbase/xbconfig.h>
#endif

#include <xbase/xbase.h>

#include <stdio.h>
#include <stdlib.h>

/*! \file index.cpp
*/

#ifdef XB_INDEX_ANY
//! Constructor
/*!
  \param pdbf
*/
xbIndex::xbIndex(xbDbf * pdbf)
{
  index          = this;
  dbf            = pdbf;
  ExpressionTree = NULL;
  indexfp        = NULL;
  IndexStatus    = 0;
  CurDbfRec      = 0L;
  KeyBuf         = NULL;
  KeyBuf2        = NULL;
#ifdef XB_LOCKING_ON
  CurLockCount   = 0;
  CurLockType    = -1;
#endif // XB_LOCKING_ON
}

void
xbIndex::Flush()
{
  if(indexfp)
    fflush(indexfp);
}

#endif // XB_INDEX_ANY

