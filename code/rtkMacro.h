/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

//#ifndef RTKMACRO_H
//#define RTKMACRO_H

#ifndef __rtkMacro_h
#define __rtkMacro_h

#include <iostream>
#include <itkMacro.h>

//--------------------------------------------------------------------
/** \brief Debugging macro, displays name and content of a variable
 *
 * \author Simon Rit
 *
 * \ingroup Macro
 */
#define DD(a) std::cout << #a " = [ " << a << " ]" << std::endl;
//--------------------------------------------------------------------

//--------------------------------------------------------------------
//--------------------------------------------------------------------
/** \brief Process gengetopt with config file option
 *
 * \author Simon Rit
 *
 * \ingroup Macro
 */
#define GGO(ggo_filename, args_info)                                    \
  args_info_##ggo_filename args_info;                                   \
  cmdline_parser_##ggo_filename##2 (argc, argv, &args_info, 1, 1, 0);   \
  if (args_info.config_given)                                           \
    cmdline_parser_##ggo_filename##_configfile (args_info.config_arg, &args_info, 0, 0, 1); \
  else cmdline_parser_##ggo_filename(argc, argv, &args_info);
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/** \brief Update a filter and catching/displaying exceptions
 *
 * \author Simon Rit
 *
 * \ingroup Macro
 */
#define TRY_AND_EXIT_ON_ITK_EXCEPTION(execFunc)                         \
  try                                                                   \
    {                                                                   \
    execFunc;                                                          \
    }                                                                   \
  catch( itk::ExceptionObject & err )                                   \
    {                                                                   \
    std::cerr << "ExceptionObject caught with " #execFunc << std::endl; \
    std::cerr << err << std::endl;                                      \
    exit(EXIT_FAILURE);                                                 \
    }
//--------------------------------------------------------------------
/** \brief Set and Get built-in type (std::vector). Creates members
 *  Get"name()" and Set"name"()
 *
 * \author Simon Rit
 *
 * \ingroup Macro
 */
#define rtkGetStdVectorMacro(name, type)                              \
  virtual type Get##name ()                                           \
  {                                                                   \
    itkDebugMacro("returning std::vector");                           \
    return this->m_##name;                                            \
  }

#define rtkSetStdVectorMacro(name, type)                              \
  virtual void Set##name (const type _arg)                            \
  {                                                                   \
    itkDebugMacro("setting " #name " to " << _arg);                   \
    if ( this->m_##name != _arg )                                     \
    {                                                                 \
      this->m_##name = _arg;                                          \
      this->Modified();                                               \
    }                                                                 \
  }

#endif // RTKMACRO_H
