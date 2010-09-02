/*  $Id: retcodes.h,v 1.5 2003/08/16 19:59:39 gkunkel Exp $

    Xbase project source code

    This file contains a listing of all the Xbase return codes.

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

/*! \file retcodes.h
*/

#ifndef __XB_RETCODES_H__
#define __XB_RETCODES_H__

/***********************************************/
/* Return Codes and Error Messages             */

#define XB_NO_ERROR                 0
#define XB_EOF                    -100
#define XB_BOF                    -101
#define XB_NO_MEMORY              -102
#define XB_FILE_EXISTS            -103
#define XB_OPEN_ERROR             -104
#define XB_WRITE_ERROR            -105
#define XB_UNKNOWN_FIELD_TYPE     -106
#define XB_ALREADY_OPEN           -107
#define XB_NOT_XBASE              -108
#define XB_INVALID_RECORD         -109
#define XB_INVALID_OPTION         -110
#define XB_NOT_OPEN               -111
#define XB_SEEK_ERROR             -112
#define XB_READ_ERROR             -113
#define XB_NOT_FOUND              -114
#define XB_FOUND                  -115
#define XB_INVALID_KEY            -116
#define XB_INVALID_NODELINK       -117
#define XB_KEY_NOT_UNIQUE         -118
#define XB_INVALID_KEY_EXPRESSION -119
#define XB_DBF_FILE_NOT_OPEN      -120
#define XB_INVALID_KEY_TYPE       -121
#define XB_INVALID_NODE_NO        -122
#define XB_NODE_FULL              -123
#define XB_INVALID_FIELDNO        -124
#define XB_INVALID_DATA           -125
#define XB_NOT_LEAFNODE           -126
#define XB_LOCK_FAILED            -127
#define XB_CLOSE_ERROR            -128
#define XB_INVALID_SCHEMA         -129
#define XB_INVALID_NAME           -130
#define XB_INVALID_BLOCK_SIZE     -131
#define XB_INVALID_BLOCK_NO       -132
#define XB_NOT_MEMO_FIELD         -133
#define XB_NO_MEMO_DATA           -134
#define XB_EXP_SYNTAX_ERROR       -135
#define XB_PARSE_ERROR            -136
#define XB_NO_DATA                -137
#define XB_UNKNOWN_TOKEN_TYPE     -138
#define XB_INVALID_FIELD          -140
#define XB_INSUFFICIENT_PARMS     -141
#define XB_INVALID_FUNCTION       -142
#define XB_INVALID_FIELD_LEN      -143
#define XB_HARVEST_NODE           -144
#define XB_INVALID_DATE           -145
#endif   /* __XB_RETCODES_H__ */
