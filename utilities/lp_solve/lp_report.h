#ifndef HEADER_lp_report
#define HEADER_lp_report

#ifdef __cplusplus
extern "C" {
#endif

/* General information functions */
const char * __VACALL explain(lprec *lp, const char *format, ...);
void __VACALL report(lprec *lp, int level, const char * format, ...);

/* Prototypes for debugging and general data dumps */
void debug_print(lprec *lp, const char *format, ...);
void debug_print_solution(lprec *lp);
void debug_print_bounds(lprec *lp, REAL *upbo, REAL *lowbo);
void blockWriteLREAL(FILE *output, const char *label, LREAL *vector, int first, int last);
void blockWriteAMAT(FILE *output, const char *label, lprec* lp, int first, int last);
void blockWriteBMAT(FILE *output, const char *label, lprec* lp, int first, int last);


/* Model reporting headers */
void REPORT_objective(lprec *lp);
void REPORT_solution(lprec *lp, int columns);
void REPORT_constraints(lprec *lp, int columns);
void REPORT_duals(lprec *lp);
void REPORT_extended(lprec *lp);

/* Other rarely used, but sometimes extremely useful reports */
void REPORT_constraintinfo(lprec *lp, const char *datainfo);
void REPORT_modelinfo(lprec *lp, MYBOOL doName, const char *datainfo);
void REPORT_lp(lprec *lp);
MYBOOL REPORT_tableau(lprec *lp);
void REPORT_scales(lprec *lp);
MYBOOL REPORT_debugdump(lprec *lp, const char *filename, MYBOOL livedata);
MYBOOL REPORT_mat_mmsave(lprec *lp, const char *filename, int *colndx, MYBOOL includeOF, const char *infotext);

#ifdef __cplusplus
 }
#endif

#endif /* HEADER_lp_report */

