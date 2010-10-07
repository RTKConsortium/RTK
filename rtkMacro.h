#ifndef RTKMACRO_H
#define RTKMACRO_H

#include <iostream>

//--------------------------------------------------------------------
#define DD(a) std::cout << #a " = [ " << a << " ]" << std::endl;
//--------------------------------------------------------------------

//--------------------------------------------------------------------
#define GGO(ggo_filename, args_info)                                    \
  args_info_##ggo_filename args_info;                                   \
  cmdline_parser_##ggo_filename##2(argc, argv, &args_info, 1, 1, 0);                    \
  if (args_info.config_given)                                           \
    cmdline_parser_##ggo_filename##_configfile (args_info.config_arg, &args_info, 0, 0, 1); \
  else cmdline_parser_##ggo_filename(argc, argv, &args_info);
//--------------------------------------------------------------------

#endif // RTKMACRO_H
