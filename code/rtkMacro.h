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

#ifndef rtkMacro_h
#define rtkMacro_h

#include <iostream>
#include <itkMacro.h>
#include <itkImageBase.h>
#include "rtkGgoArgsInfoManager.h"

//--------------------------------------------------------------------
#ifndef CLANG_PRAGMA_PUSH
#define ITK_PRAGMA(x) _Pragma (#x)
#if defined(__clang__) && defined(__has_warning)
#define CLANG_PRAGMA_PUSH ITK_PRAGMA(clang diagnostic push)
#define CLANG_PRAGMA_POP  ITK_PRAGMA(clang diagnostic pop)
# if __has_warning("-Wfloat-equal")
#define CLANG_SUPPRESS_Wfloat_equal ITK_PRAGMA( clang diagnostic ignored "-Wfloat-equal" )
# endif
#else
#define CLANG_PRAGMA_PUSH
#define CLANG_PRAGMA_POP
#define CLANG_SUPPRESS_Wfloat_equal
#endif
#endif
//--------------------------------------------------------------------

//--------------------------------------------------------------------
#ifndef ITK_NULLPTR
# define ITK_NULLPTR NULL
#endif
//--------------------------------------------------------------------

//--------------------------------------------------------------------
#ifndef ITK_OVERRIDE
# define ITK_OVERRIDE
#endif
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/** \brief Debugging macro, displays name and content of a variable
 *
 * \author Simon Rit
 *
 * \ingroup Macro
 */
#ifndef DD
#  define DD(a) std::cout << #a " = [ " << a << " ]" << std::endl;
#endif
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/** \brief Process gengetopt with config file option
 *
 * \author Simon Rit
 *
 * \ingroup Macro
 */
#define GGO(ggo_filename, args_info)                                                                       \
  args_info_##ggo_filename args_info;                                                                      \
  cmdline_parser_##ggo_filename##_params args_params;                                                      \
  cmdline_parser_##ggo_filename##_params_init(&args_params);                                               \
  args_params.print_errors = 1;                                                                            \
  args_params.check_required = 0;                                                                          \
  args_params.override = 1;                                                                                \
  args_params.initialize = 1;                                                                              \
  if(0 != cmdline_parser_##ggo_filename##_ext(argc, argv, &args_info, &args_params) )                      \
    {                                                                                                      \
    std::cerr << "Error in cmdline_parser_" #ggo_filename "_ext" << std::endl;                             \
    exit(1);                                                                                               \
    }                                                                                                      \
  std::string configFile;                                                                                  \
  if(args_info.config_given)                                                                               \
    configFile = args_info.config_arg;                                                                     \
  cmdline_parser_##ggo_filename##_free(&args_info);                                                        \
  if (configFile != "")                                                                                    \
    {                                                                                                      \
    if(0 != cmdline_parser_##ggo_filename##_config_file (configFile.c_str(), &args_info, &args_params) )   \
      {                                                                                                    \
      std::cerr << "Error in cmdline_parser_" #ggo_filename "_config_file" << std::endl;                   \
      exit(1);                                                                                             \
      }                                                                                                    \
    args_params.initialize = 0;                                                                            \
    }                                                                                                      \
  args_params.check_required = 1;                                                                          \
  if(0 != cmdline_parser_##ggo_filename##_ext(argc, argv, &args_info, &args_params) )                      \
    {                                                                                                      \
    std::cerr << "Error in cmdline_parser_" #ggo_filename "_ext" << std::endl;                             \
    exit(1);                                                                                               \
    }                                                                                                      \
  rtk::args_info_manager< args_info_##ggo_filename >                                                       \
     manager_object( args_info, cmdline_parser_##ggo_filename##_free );
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
    execFunc;                                                           \
    }                                                                   \
  catch( itk::ExceptionObject & err )                                   \
    {                                                                   \
    std::cerr << "ExceptionObject caught with " #execFunc               \
              << " in file " << __FILE__                                \
              << " line " << __LINE__                                   \
              << std::endl;                                             \
    std::cerr << err << std::endl;                                      \
    itk::InvalidRequestedRegionError* r;                                \
    r = dynamic_cast<itk::InvalidRequestedRegionError*>(&err);          \
    if(r)                                                               \
      {                                                                 \
      if( r->GetDataObject()->GetSource() )                             \
        {                                                               \
        std::cerr << "Invalid requested region error triggered by "     \
                  << r->GetDataObject()->GetSource()->GetNameOfClass()  \
                  << std::endl;                                         \
      }                                                                 \
      itk::ImageBase<3> *img;                                           \
      img = dynamic_cast<itk::ImageBase<3>*>(r->GetDataObject());       \
      if(img)                                                           \
        {                                                               \
        DD(img->GetRequestedRegion())                                   \
        DD(img->GetLargestPossibleRegion())                             \
        }                                                               \
      }                                                                 \
      exit(EXIT_FAILURE);                                               \
    }
//--------------------------------------------------------------------

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
    itkDebugMacro("setting " #name " of type " #type );               \
    if ( this->m_##name != _arg )                                     \
    {                                                                 \
      this->m_##name = _arg;                                          \
      this->Modified();                                               \
    }                                                                 \
  }
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/** \brief Redefine ITK's New macros in order to add a watcher to
 * each new filter created
 *
 * \author Cyril Mory
 *
 * \ingroup Macro
 */

#ifdef RTK_TIME_EACH_FILTER
#undef itkSimpleNewMacro
#define itkSimpleNewMacro(x)                                                         \
  static Pointer New(void)                                                           \
    {                                                                                \
    Pointer smartPtr = ::itk::ObjectFactory< x >::Create();                          \
    if ( smartPtr.GetPointer() == ITK_NULLPTR )                                      \
      {                                                                              \
      smartPtr = new x;                                                              \
      }                                                                              \
    smartPtr->UnRegister();                                                          \
    /* If smartPtr is a ProcessObject, watch it */                                   \
    itk::ProcessObject* processObjectPointer = ITK_NULLPTR;                                 \
    processObjectPointer = dynamic_cast<itk::ProcessObject*>(smartPtr.GetPointer()); \
    if (processObjectPointer != ITK_NULLPTR)                                                \
      {                                                                              \
      rtk::GlobalTimer::GetInstance()->Watch(processObjectPointer);                  \
      }                                                                              \
    return smartPtr;                                                                 \
    }

#undef itkCreateAnotherMacro
#define itkCreateAnotherMacro(x)                                            \
  virtual::itk::LightObject::Pointer CreateAnother(void) const ITK_OVERRIDE \
    {                                                                       \
    ::itk::LightObject::Pointer smartPtr;                                   \
    smartPtr = x::New().GetPointer();                                       \
    return smartPtr;                                                        \
    }

#undef itkFactorylessNewMacro
#define itkFactorylessNewMacro(x)                                                    \
  static Pointer New(void)                                                           \
    {                                                                                \
    Pointer smartPtr;                                                                \
    x *     rawPtr = new x;                                                          \
    smartPtr = rawPtr;                                                               \
    rawPtr->UnRegister();                                                            \
    /* If smartPtr is a ProcessObject, watch it */                                   \
    itk::ProcessObject* processObjectPointer = ITK_NULLPTR;                                 \
    processObjectPointer = dynamic_cast<itk::ProcessObject*>(smartPtr.GetPointer()); \
    if (processObjectPointer != ITK_NULLPTR)                                                \
      {                                                                              \
      rtk::GlobalTimer::GetInstance()->Watch(processObjectPointer);                  \
      }                                                                              \
    return smartPtr;                                                                 \
    }                                                                                \
  virtual::itk::LightObject::Pointer CreateAnother(void) const ITK_OVERRIDE          \
    {                                                                                \
    ::itk::LightObject::Pointer smartPtr;                                            \
    smartPtr = x::New().GetPointer();                                                \
    return smartPtr;                                                                 \
    }
#endif //RTK_TIME_EACH_FILTER


#endif
