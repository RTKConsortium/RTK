/*
This file is licensed to you under the license specified in the included file
`LICENSE'. Look there for further details.
*/


/*
  Called by yyparse on error.
 */

#include "yyerror.h"

#include <stdio.h>
#include <string.h>
#include <iostream>

extern int gengetopt_count_line;
extern char * gengetopt_input_filename;

extern int tokenpos;
extern char linebuf[];
extern char *yytext;

using namespace std;

void
yyerror (const char *s)
{
  const char *source =
    (gengetopt_input_filename ? gengetopt_input_filename : "gengetopt");

  fprintf (stderr, "%s:%d: %s %s\n", source, gengetopt_count_line, s, yytext);

  if (/*!linebuf || */!strlen(linebuf))
    return;

  fprintf (stderr, "%s:%d: %s\n", source, gengetopt_count_line, linebuf);
  fprintf (stderr, "%s:%d: %*s\n", source, gengetopt_count_line,
           tokenpos + 1, "^");
}

void
yyerror (gengetopt_option *opt, const char *s)
{
  const char *source =
    (opt->filename ? opt->filename : "gengetopt");

  cerr << source << ":" << opt->linenum << ": " << s << endl;
}
