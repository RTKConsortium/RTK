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
#ifndef __srtkCreateKernel_h
#define __srtkCreateKernel_h


#include "srtkKernel.h"
#include <itkFlatStructuringElement.h>

namespace rtk
{

namespace simple
{

#define srtkKernelPolygonCreateMacro(n) \
  case srtkPolygon##n: return ITKKernelType::Polygon( radius, n )

template< unsigned int VImageDimension >
itk::FlatStructuringElement< VImageDimension >
CreateKernel( KernelEnum kernelType, const std::vector<uint32_t> &size )
{
  typedef typename itk::FlatStructuringElement< VImageDimension > ITKKernelType;

  typename ITKKernelType::SizeType radius = srtkSTLVectorToITK<typename ITKKernelType::SizeType>( size );

  switch (kernelType)
    {
    case srtkAnnulus:
      return ITKKernelType::Annulus( radius, 1, false );
    case srtkBall:
      return ITKKernelType::Ball( radius );
    case srtkBox:
      return ITKKernelType::Box( radius );
    case srtkCross:
      return ITKKernelType::Cross( radius );
    srtkKernelPolygonCreateMacro(3);
    srtkKernelPolygonCreateMacro(4);
    srtkKernelPolygonCreateMacro(5);
    srtkKernelPolygonCreateMacro(6);
    srtkKernelPolygonCreateMacro(7);
    srtkKernelPolygonCreateMacro(8);
    srtkKernelPolygonCreateMacro(9);
    default:
      srtkExceptionMacro( "Logic Error: Unknown Kernel Type" );
    }

#undef srtkKernelPolygonCreateMacro


}


} // end namespace simple
} // end namespace rtk


#endif //__srtkCreateKernel_h
