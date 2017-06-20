/*=========================================================================
 *
 *  Copyright Insight Software Consortium & RTK Consortium
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

#if defined(_MSC_VER) && _MSC_VER == 1700
// VS11 (Visual Studio 2012) has limited variadic argument support for
// std::bind, the following increases the number of supported
// arguments.
// https://connect.microsoft.com/VisualStudio/feedback/details/723448/very-few-function-arguments-for-std-bind

 #if defined(_VARIADIC_MAX) && _VARIADIC_MAX < 10
  #error "_VARIADIC_MAX already defined. Some STL classes may have insufficient number of template parameters."
 #else
  #define _VARIADIC_MAX 10
 #endif
#endif


#include "srtkImage.hxx"
#include "srtkMemberFunctionFactory.h"

namespace rtk
{
  namespace simple
  {
    void Image::Allocate ( unsigned int Width, unsigned int Height, unsigned int Depth, unsigned int dim4, PixelIDValueEnum ValueEnum, unsigned int numberOfComponents )
    {
      // initialize member function factory for allocating images

      // The pixel IDs supported
      typedef AllPixelIDTypeList              PixelIDTypeList;

      typedef void ( Self::*MemberFunctionType )( unsigned int, unsigned int, unsigned int, unsigned int, unsigned int );

      typedef AllocateMemberFunctionAddressor< MemberFunctionType > AllocateAddressor;

      detail::MemberFunctionFactory< MemberFunctionType > allocateMemberFactory( this );
      allocateMemberFactory.RegisterMemberFunctions< PixelIDTypeList, 2, AllocateAddressor > ();
      allocateMemberFactory.RegisterMemberFunctions< PixelIDTypeList, 3, AllocateAddressor > ();
      allocateMemberFactory.RegisterMemberFunctions< PixelIDTypeList, 4, AllocateAddressor > ();

      if ( ValueEnum == srtkUnknown )
        {
        srtkExceptionMacro( "Unable to construct image of unsupported pixel type" );
        }

      if ( Depth == 0 )
        {
        allocateMemberFactory.GetMemberFunction( ValueEnum, 2 )( Width, Height, Depth, dim4, numberOfComponents );
        }
      else if ( dim4 == 0 )
        {
        allocateMemberFactory.GetMemberFunction( ValueEnum, 3 )( Width, Height, Depth, dim4, numberOfComponents );
        }
      else
        {
        allocateMemberFactory.GetMemberFunction( ValueEnum, 4 )( Width, Height, Depth, dim4, numberOfComponents );
        }
    }
  }
}


//
// There is only one templated function in the external interface
// which need to be instantiated, so that the itk::Image and the
// srtk::PimpleImage are completely encapsulated. That is the
// InternalInitialization method. The following uses a macro to
// explicitly instantiate for the expected image types.
//

#define SRTK_TEMPLATE_InternalInitialization_D( _I, _D )                \
  namespace rtk { namespace simple {                                    \
  template SRTKCommon_EXPORT void Image::InternalInitialization<_I,_D>(  PixelIDToImageType< typelist::TypeAt<InstantiatedPixelIDTypeList, \
                                                                                            _I>::Result, \
                                                                           _D>::ImageType *i ); \
  } }

#ifdef SRTK_4D_IMAGES
#define SRTK_TEMPLATE_InternalInitialization( _I ) SRTK_TEMPLATE_InternalInitialization_D( _I, 2 ) SRTK_TEMPLATE_InternalInitialization_D( _I, 3 ) SRTK_TEMPLATE_InternalInitialization_D( _I, 4 )
#else
#define SRTK_TEMPLATE_InternalInitialization( _I ) SRTK_TEMPLATE_InternalInitialization_D( _I, 2 ) SRTK_TEMPLATE_InternalInitialization_D( _I, 3 )
#endif


// Instantiate for all types in the lists
SRTK_TEMPLATE_InternalInitialization( 0 );
SRTK_TEMPLATE_InternalInitialization( 1 );
SRTK_TEMPLATE_InternalInitialization( 2 );
SRTK_TEMPLATE_InternalInitialization( 3 );
SRTK_TEMPLATE_InternalInitialization( 4 );
SRTK_TEMPLATE_InternalInitialization( 5 );
SRTK_TEMPLATE_InternalInitialization( 6 );
SRTK_TEMPLATE_InternalInitialization( 7 );
SRTK_TEMPLATE_InternalInitialization( 8 );
SRTK_TEMPLATE_InternalInitialization( 9 );
SRTK_TEMPLATE_InternalInitialization( 10 );
SRTK_TEMPLATE_InternalInitialization( 11 );
SRTK_TEMPLATE_InternalInitialization( 12 );
SRTK_TEMPLATE_InternalInitialization( 13 );
SRTK_TEMPLATE_InternalInitialization( 14 );
SRTK_TEMPLATE_InternalInitialization( 15 );
SRTK_TEMPLATE_InternalInitialization( 16 );
SRTK_TEMPLATE_InternalInitialization( 17 );
SRTK_TEMPLATE_InternalInitialization( 18 );
SRTK_TEMPLATE_InternalInitialization( 19 );
SRTK_TEMPLATE_InternalInitialization( 20 );
SRTK_TEMPLATE_InternalInitialization( 21 );
SRTK_TEMPLATE_InternalInitialization( 22 );
SRTK_TEMPLATE_InternalInitialization( 23 );
SRTK_TEMPLATE_InternalInitialization( 24 );
SRTK_TEMPLATE_InternalInitialization( 25 );
SRTK_TEMPLATE_InternalInitialization( 26 );
SRTK_TEMPLATE_InternalInitialization( 27 );
SRTK_TEMPLATE_InternalInitialization( 28 );
SRTK_TEMPLATE_InternalInitialization( 29 );


srtkStaticAssert( typelist::Length<rtk::simple::InstantiatedPixelIDTypeList>::Result < 30, "Number of explicitly instantiated pixel types is more then expected!" );
