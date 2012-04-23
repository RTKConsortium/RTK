#ifndef RTKMACRO_H
#define RTKMACRO_H

#include <iostream>

//--------------------------------------------------------------------
#define DD(a) std::cout << #a " = [ " << a << " ]" << std::endl;
//--------------------------------------------------------------------

//--------------------------------------------------------------------
#define GGO(ggo_filename, args_info)                                    \
  args_info_##ggo_filename args_info;                                   \
  cmdline_parser_##ggo_filename##2 (argc, argv, &args_info, 1, 1, 0);   \
  if (args_info.config_given)                                           \
    cmdline_parser_##ggo_filename##_configfile (args_info.config_arg, &args_info, 0, 0, 1); \
  else cmdline_parser_##ggo_filename(argc, argv, &args_info);
//--------------------------------------------------------------------

//--------------------------------------------------------------------
#define TRY_AND_EXIT_ON_ITK_EXCEPTION(execFunc)                         \
  try                                                                   \
    {                                                                   \
    execFunc ;                                                          \
    }                                                                   \
  catch( itk::ExceptionObject & err )                                   \
    {                                                                   \
    std::cerr << "ExceptionObject caught with " #execFunc << std::endl; \
    std::cerr << err << std::endl;                                      \
    exit(EXIT_FAILURE);                                                 \
    }
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/** Set built-in type. Creates member Set"name"() ;                     */
#define rtkSetMacro(name,type)                                          \
  virtual void Set##name (const type _arg)                              \
  {                                                                     \
    if (this->m_##name != _arg)                                         \
      {                                                                 \
      this->m_##name = _arg;                                            \
      this->Modified();                                                 \
      }                                                                 \
  }
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/** Get built-in type.  Creates member Get"name"() (e.g., GetVisibility()); */
#define rtkGetMacro(name,type)                                          \
  virtual type Get##name ()                                             \
  {                                                                     \
    return this->m_##name;                                              \
  }
//--------------------------------------------------------------------

#endif // RTKMACRO_H
