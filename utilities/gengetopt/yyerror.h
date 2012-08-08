#ifndef YYERROR_H_
#define YYERROR_H_

#include "ggos.h"

void
yyerror (const char *s);

void
yyerror (gengetopt_option *opt, const char *s);

#endif /*YYERROR_H_*/
