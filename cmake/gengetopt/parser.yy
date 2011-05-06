/**
 * Copyright (C) 1999-2007  Free Software Foundation, Inc.
 *
 * This file is part of GNU gengetopt
 *
 * GNU gengetopt is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU gengetopt is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with gengetopt; see the file COPYING. If not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */


%{

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string>

#include "my_sstream.h"

#include "acceptedvalues.h"

#include "argsdef.h"

#include "gengetopt.h"
#include "errorcodes.h"
#include "ggos.h"
#include "yyerror.h"

extern int gengetopt_count_line;
extern char * gengetopt_input_filename;

static int gengetopt_package_given = 0;
static int gengetopt_version_given = 0;
static int gengetopt_purpose_given = 0;
static int gengetopt_usage_given = 0;
static int gengetopt_description_given = 0;

/// the last option parsed
static gengetopt_option *current_option = 0;

extern int yylex (void) ;

//#define YYERROR_VERBOSE 1

void check_result(int o, gengetopt_option *opt)
{
  if (o)
    {
        ostringstream err;

    switch (o)
    {
    case NOT_ENOUGH_MEMORY:
        yyerror (opt, "not enough memory");
    	break;
    case REQ_LONG_OPTION:
        err << "long option redefined \'" << opt->long_opt << "\'";
        yyerror (opt, err.str().c_str());
		break;
    case REQ_SHORT_OPTION:
        err << "short option redefined \'" << opt->short_opt << "\'";
        yyerror (opt, err.str().c_str());
        break;
    case FOUND_BUG:
        yyerror (opt, "bug found!!");
        break;
    case GROUP_UNDEFINED:
        yyerror (opt, "group undefined");
        break;
    case MODE_UNDEFINED:
        yyerror (opt, "mode undefined");
        break;
    case INVALID_DEFAULT_VALUE:
        yyerror (opt, "invalid default value");
        break;
    case NOT_REQUESTED_TYPE:
        yyerror (opt, "type specification not requested");
        break;
    case NOT_VALID_SPECIFICATION:
      yyerror (opt, "invalid specification for this kind of option");
      break;
    case SPECIFY_FLAG_STAT:
      yyerror (opt, "you must specify the default flag status");
      break;
    case NOT_GROUP_OPTION:
      yyerror (opt, "group specification for a non group option");
      break;
    case NOT_MODE_OPTION:
      yyerror (opt, "mode specification for an option not belonging to a mode");
      break;
    case SPECIFY_GROUP:
      yyerror (opt, "missing group specification");
      break;
    case SPECIFY_MODE:
      yyerror (opt, "missing mode specification");
      break;
    case INVALID_NUMERIC_VALUE:
        yyerror (opt, "invalid numeric value");
        break;
    case INVALID_ENUM_TYPE_USE:
    	yyerror (opt, "enum type can only be specified for options with values");
        break;
    case HELP_REDEFINED:
    	yyerror (opt, "if you want to redefine --help, please use option --no-help");
        break;
    case VERSION_REDEFINED:
    	yyerror (opt, "if you want to redefine --version, please use option --no-version");
        break;
    }
  }
}

/* the number of allowed occurrences of a multiple option */
struct multiple_size
{
    /* these strings are allocated dynamically and NOT
      automatically freed upon destruction */
    char *min;
    char *max;

    /* if no limit is specified then initialized to 0.
       if the same size is specified for min and max, it means that an exact
       number of occurrences is required*/
    multiple_size(const char *m = "0", const char *M = "0") :
        min(strdup(m)), max(strdup(M))
    {}
};

#define check_error if (o) YYERROR;

%}

%union {
    char   *str;
    char    chr;
    int	    argtype;
    int	    boolean;
    class AcceptedValues *ValueList;
    struct gengetopt_option *gengetopt_option;
    struct multiple_size *multiple_size;
}

%token		    TOK_PACKAGE		"package"
%token              TOK_VERSION		"version"
%token              TOK_OPTION		"option"
%token              TOK_DEFGROUP	"defgroup"
%token              TOK_GROUPOPTION	"groupoption"
%token              TOK_DEFMODE		"defmode"
%token              TOK_MODEOPTION	"modeoption"
%token              TOK_YES		"yes"
%token              TOK_NO		"no"
%token              TOK_ON		"on"
%token              TOK_OFF		"off"
%token              TOK_FLAG		"flag"
%token              TOK_PURPOSE		"purpose"
%token              TOK_DESCRIPTION	"description"
%token              TOK_USAGE		"usage"
%token              TOK_DEFAULT		"default"
%token              TOK_GROUP		"group"
%token              TOK_GROUPDESC	"groupdesc"
%token              TOK_MODE		"mode"
%token              TOK_MODEDESC	"modedesc"
%token              TOK_MULTIPLE	"multiple"
%token              TOK_ARGOPTIONAL	"argoptional"
%token              TOK_TYPESTR		"typestr"
%token              TOK_SECTION		"section"
%token              TOK_DETAILS		"details"
%token              TOK_SECTIONDESC	"sectiondesc"
%token              TOK_TEXT    	"text"
%token              TOK_ARGS    	"args"
%token              TOK_VALUES          "values"
%token              TOK_HIDDEN      "hidden"
%token              TOK_DEPENDON      "dependon"
%token <str>        TOK_STRING
%token <chr>        TOK_CHAR
%token <argtype>    TOK_ARGTYPE
%token <str>        TOK_SIZE

%type  <boolean>    req_onoff
%type  <boolean>    opt_yesno optional_yesno
%type  <str>        quoted_string
%type  <str>        opt_groupdesc
%type  <str>        opt_sectiondesc
%type  <str>        opt_modedesc
%type  <ValueList>  listofvalues
%type  <str>        acceptedvalue
%type  <gengetopt_option> option_parts
%type  <multiple_size> multiple_size


%% /* ====================================================================== */


input
	: /* empty */
	| statement input
	;


statement
	: package
	| version
	| args
	| purpose
	| description
	| usage
	| sectiondef
	| option
	| text
	| groupoption
	| groupdef
	| modeoption
	| modedef
	;


package
	: TOK_PACKAGE TOK_STRING
	    {
	      if (gengetopt_package_given)
		{
		  yyerror ("package redefined");
		  YYERROR;
		}
	      else
		{
		  gengetopt_package_given = 1;
		  if (gengetopt_define_package ($2))
		    {
		      yyerror ("not enough memory");
		      YYERROR;
		    }
		}
	    }
	;

version
	: TOK_VERSION TOK_STRING
	    {
	      if (gengetopt_version_given)
		{
		  yyerror ("version redefined");
		  YYERROR;
		}
	      else
		{
		  gengetopt_version_given = 1;
		  if (gengetopt_define_version ($2))
		    {
		      yyerror ("not enough memory");
		      YYERROR;
		    }
		}
	    }
	;

purpose
	: TOK_PURPOSE quoted_string
	    {
	      if (gengetopt_purpose_given)
		{
		  yyerror ("purpose redefined");
		  YYERROR;
		}
	      else
		{
		  gengetopt_purpose_given = 1;
		  if (gengetopt_define_purpose ($2))
		    {
		      yyerror ("not enough memory");
		      YYERROR;
		    }
		}
	    }
	;

description
	: TOK_DESCRIPTION quoted_string
	    {
	      if (gengetopt_description_given)
		{
		  yyerror ("description redefined");
		  YYERROR;
		}
	      else
		{
		  gengetopt_description_given = 1;
		  if (gengetopt_define_description ($2))
		    {
		      yyerror ("not enough memory");
		      YYERROR;
		    }
		}
	    }
	;

usage
  : TOK_USAGE quoted_string
  {
      if (gengetopt_usage_given)
      {
	  yyerror ("usage redefined");
	  YYERROR;
      }
      else
      {
	  gengetopt_usage_given = 1;
	  if (gengetopt_define_usage ($2))
          {
	      yyerror ("not enough memory");
	      YYERROR;
          }
      }
  }
        ;


sectiondef
          : TOK_SECTION quoted_string opt_sectiondesc
              {
                gengetopt_set_section ($2, $3);
              }
          ;

text
  : TOK_TEXT quoted_string
            {
            	if (current_option) {
            		std::string current_option_text;
            		if (current_option->text_after) {
            			current_option_text = std::string(current_option->text_after) + $2;
            			current_option->text_after = strdup(current_option_text.c_str()); 
            		} else {
	            		current_option->text_after = strdup($2);
	            	}
            	} else {
					gengetopt_set_text($2);
  				}
            }
        ;

args
  : TOK_ARGS TOK_STRING
            {
  gengetopt_set_args($2);
            }
        ;

groupdef
	: TOK_DEFGROUP TOK_STRING opt_groupdesc optional_yesno
	    {
              if (gengetopt_add_group ($2, $3, $4))
                {
		  	yyerror ("group redefined");
		  	YYERROR;
		  }
	    }
	;

modedef
	: TOK_DEFMODE TOK_STRING opt_modedesc
	    {
              if (gengetopt_add_mode ($2, $3))
                {
		  	yyerror ("mode redefined");
		  	YYERROR;
		  }
	    }
	;

option
	: TOK_OPTION TOK_STRING TOK_CHAR quoted_string
		option_parts
	    {
          $5->filename = gengetopt_input_filename;
          $5->linenum = @1.first_line;
	      $5->long_opt = strdup($2);
	      if ($3 != '-')
	      	$5->short_opt = $3;
	      $5->desc = strdup($4);
	      int o = gengetopt_check_option ($5, false);
	      check_result(o, $5);
          check_error;
	      o = gengetopt_add_option ($5);
	      check_result(o, $5);
	      check_error;
	      current_option = $5;
	    }
	;

groupoption
	: TOK_GROUPOPTION TOK_STRING TOK_CHAR quoted_string
                option_parts
	    {
          $5->filename = gengetopt_input_filename;
          $5->linenum = @1.first_line;
	      $5->long_opt = strdup($2);
          if ($3 != '-')
            $5->short_opt = $3;
          $5->desc = strdup($4);
          int o = gengetopt_check_option ($5, true);
          check_result(o, $5);
          check_error;
          o = gengetopt_add_option ($5);
          check_result(o, $5);
          check_error;
	    }
        ;

modeoption
	: TOK_MODEOPTION TOK_STRING TOK_CHAR quoted_string
                option_parts
	    {
          $5->filename = gengetopt_input_filename;
          $5->linenum = @1.first_line;
	      $5->long_opt = strdup($2);
          if ($3 != '-')
            $5->short_opt = $3;
          $5->desc = strdup($4);
          int o = gengetopt_check_option ($5, false, true);
          check_result(o, $5);
          check_error;
          o = gengetopt_add_option ($5);
          check_result(o, $5);
          check_error;
	    }
        ;


/* ---------------------------------------------------------------------- */

quoted_string
	: TOK_STRING
	;

option_parts: option_parts opt_yesno
			  {
			  	$$ = $1;
			  	$$->required = $2;
			  	$$->required_set = true;
			  }
			| option_parts TOK_ARGTYPE
			  {
			  	$$ = $1;
			  	$$->type = $2;
			  }
			| option_parts TOK_TYPESTR '=' TOK_STRING
			  {
			  	$$ = $1;
			  	$$->type_str = strdup($4);
			  }
			| option_parts TOK_DETAILS '=' quoted_string
			  {
			  	$$ = $1;
			  	$$->details = strdup($4);
			  }
			| option_parts TOK_VALUES '=' listofvalues
			  {
			  	$$ = $1;
			  	$$->acceptedvalues = $4;
			  }
			| option_parts TOK_DEFAULT '=' TOK_STRING
			  {
			  	$$ = $1;
			  	$$->default_string = strdup($4);
			  }
            | option_parts TOK_GROUP '=' TOK_STRING
              {
                $$ = $1;
                $$->group_value = strdup($4);
              }
            | option_parts TOK_MODE '=' TOK_STRING
              {
                $$ = $1;
                $$->mode_value = strdup($4);
              }
            | option_parts TOK_DEPENDON '=' TOK_STRING
              {
                $$ = $1;
                $$->dependon = strdup($4);
              }
			| option_parts TOK_ARGOPTIONAL
			  {
			  	$$ = $1;
			  	$$->arg_is_optional = true;
			  }
			| option_parts TOK_MULTIPLE multiple_size
			  {
			  	$$ = $1;
			  	$$->multiple = true;
                $$->multiple_min = $3->min;
                $$->multiple_max = $3->max;
                delete $3;
			  }
      | option_parts TOK_FLAG
        {
          $$ = $1;
          $$->type = ARG_FLAG;
        }
      | option_parts TOK_HIDDEN
        {
          $$ = $1;
          $$->hidden = true;
        }
      | option_parts req_onoff
        {
          $$ = $1;
          $$->flagstat = $2;
        }
      | { $$ = new gengetopt_option; }
      ;

req_onoff
	: TOK_ON	{ $$ = 1; }
	| TOK_OFF	{ $$ = 0; }
	;

optional_yesno
	: /* empty */	{ $$ = 0; }
	| TOK_YES	{ $$ = 1; }
	| TOK_NO	{ $$ = 0; }
	;

opt_yesno
    : TOK_YES   { $$ = 1; }
    | TOK_NO    { $$ = 0; }
    ;

opt_groupdesc
	: /* empty */			{ $$ = 0; }
        | TOK_GROUPDESC '=' TOK_STRING	{ $$ = $3; }
	;

opt_modedesc
	: /* empty */			{ $$ = 0; }
        | TOK_MODEDESC '=' TOK_STRING	{ $$ = $3; }
	;

opt_sectiondesc
        : /* empty */			{ $$ = 0; }
        | TOK_SECTIONDESC '=' TOK_STRING	{ $$ = $3; }
        ;

listofvalues
        : acceptedvalue { $$ = new AcceptedValues; $$->insert($1); }
        | listofvalues ',' acceptedvalue { $1->insert($3); $$ = $1; }
        ;

acceptedvalue
        : TOK_STRING { $$ = $1; }
        ;

multiple_size
    : { $$ = new multiple_size; }
    | '(' TOK_SIZE ')' { $$ = new multiple_size($2, $2); }
    | '(' TOK_SIZE '-' ')' { $$ = new multiple_size($2, "0"); free($2); }
    | '(' '-' TOK_SIZE  ')' { $$ = new multiple_size("0", $3); free($3); }
    | '(' TOK_SIZE '-' TOK_SIZE  ')' { $$ = new multiple_size($2, $4); free($2); free($4); }
    ;

%%
