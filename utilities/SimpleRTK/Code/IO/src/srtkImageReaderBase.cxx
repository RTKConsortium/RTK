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
#include "srtkImageReaderBase.h"
#include "srtkMacro.h"
#include "srtkExceptionObject.h"


// Include the Transform IO here, so that the IO factory registration
// will occour.
#include <itkTransformFileReader.h>
#include <itkTransformFileWriter.h>

#include <stdint.h>

#include <string>

#include <rtkConfiguration.h>
#ifdef RTK_USE_CUDA
# include <itkCudaImage.h>
#endif
#include <itkImage.h>
#include <itkImageIOBase.h>
#include <itkImageIOFactory.h>
#include <itkGDCMImageIO.h>
#include <itksys/SystemTools.hxx>

namespace rtk {
namespace simple {

ImageReaderBase
::ImageReaderBase()
  : m_OutputPixelType(srtkUnknown),
    m_LoadPrivateTags(false)
{
}

std::string
ImageReaderBase
::ToString() const
{
  std::ostringstream out;
  out << "  OutputPixelType: ";
  this->ToStringHelper(out, this->m_OutputPixelType) << std::endl;
  out << "  LoadPrivateTags: ";
  this->ToStringHelper(out, this->m_LoadPrivateTags) << std::endl;
  out << ProcessObject::ToString();
  return out.str();
}

itk::SmartPointer<itk::ImageIOBase>
ImageReaderBase
::GetImageIOBase(const std::string &fileName)
{
  itk::ImageIOBase::Pointer iobase =
    itk::ImageIOFactory::CreateImageIO( fileName.c_str(), itk::ImageIOFactory::ReadMode);


  if ( iobase.IsNull() )
     {
     if ( !itksys::SystemTools::FileExists( fileName.c_str() ) )
       {
       srtkExceptionMacro( "The file \"" << fileName << "\" does not exist." );
       }

     if ( !bool(std::ifstream( fileName.c_str() )) )
       {
       srtkExceptionMacro( "Unable to open \"" << fileName << "\" for reading." );
       }

     srtkExceptionMacro( "Unable to determine ImageIO reader for \"" << fileName << "\"" );
    }

  // Try additional parameters
  itk::GDCMImageIO *ioGDCMImage = dynamic_cast<itk::GDCMImageIO*>(iobase.GetPointer());
  if (ioGDCMImage)
    {
    ioGDCMImage->SetLoadPrivateTags(this->m_LoadPrivateTags);
    }

  // Read the image information
  iobase->SetFileName( fileName );
  iobase->ReadImageInformation();


  return iobase;
}

ImageReaderBase::Self&
ImageReaderBase
::SetOutputPixelType( PixelIDValueEnum pixelID )
{
  this->m_OutputPixelType = pixelID;
  return *this;
}

PixelIDValueEnum
ImageReaderBase
::GetOutputPixelType( void ) const
{
  return this->m_OutputPixelType;
}


ImageReaderBase::Self&
ImageReaderBase
::SetLoadPrivateTags(bool loadPrivateTags)
{
  this->m_LoadPrivateTags = loadPrivateTags;
  return *this;
}

bool
ImageReaderBase
::GetLoadPrivateTags() const
{
  return this->m_LoadPrivateTags;
}

void
ImageReaderBase
::LoadPrivateTagsOn()
{
  this->SetLoadPrivateTags(true);
}

void
ImageReaderBase
::LoadPrivateTagsOff()
{
  this->SetLoadPrivateTags(false);
}


void
ImageReaderBase
::GetPixelIDFromImageIO( const std::string &fileName,
                         PixelIDValueType &outPixelType,
                         unsigned int & outDimensions )
{
  itk::ImageIOBase::Pointer iobase = this->GetImageIOBase(fileName);


  this->GetPixelIDFromImageIO(iobase, outPixelType, outDimensions);
}

void
ImageReaderBase
::GetPixelIDFromImageIO( itk::ImageIOBase *iobase,
                         PixelIDValueType &outPixelType,
                         unsigned int & outDimensions )
{

  // get output information about input image
  unsigned int dimension = iobase->GetNumberOfDimensions();
  itk::ImageIOBase::IOComponentType componentType = iobase->GetComponentType();
  itk::ImageIOBase::IOPixelType pixelType = iobase->GetPixelType();
  unsigned int numberOfComponents = iobase->GetNumberOfComponents();

  outDimensions = dimension;


  if (numberOfComponents == 1 &&
      ( pixelType == itk::ImageIOBase::SCALAR || pixelType == itk::ImageIOBase::COMPLEX ) )
    {
    outPixelType = this->ExecuteInternalReadScalar( componentType );
    return;
    }
  // we try to load anything else into a VectorImage
  else if  (pixelType == itk::ImageIOBase::RGB ||
            pixelType == itk::ImageIOBase::RGBA ||
            pixelType == itk::ImageIOBase::VECTOR ||
            pixelType == itk::ImageIOBase::COVARIANTVECTOR ||
            pixelType == itk::ImageIOBase::FIXEDARRAY ||
            pixelType == itk::ImageIOBase::POINT ||
            pixelType == itk::ImageIOBase::OFFSET )
    {
    outPixelType = this->ExecuteInternalReadVector( componentType );
    return;
    }
  else if ( pixelType == itk::ImageIOBase::COMPLEX )
    {
    outPixelType = this->ExecuteInternalReadComplex( componentType );
    return;
    }
  else
    {
    srtkExceptionMacro(  "Unknown PixelType: "  << (int) componentType );
    }

  srtkExceptionMacro( "Unable to load image." );
}

unsigned int
ImageReaderBase
::GetDimensionFromImageIO( itk::ImageIOBase* iobase, unsigned int i)
{
  return iobase->GetDimensions(i);
}


unsigned int
ImageReaderBase
::GetDimensionFromImageIO(const std::string &filename, unsigned int i)
{
  itk::ImageIOBase::Pointer iobase = this->GetImageIOBase(filename);

  return this->GetDimensionFromImageIO(iobase.GetPointer(), i);
}


PixelIDValueType
ImageReaderBase
::ExecuteInternalReadScalar( int componentType )
{
  const unsigned int UnusedDimension = 2;

#ifdef RTK_USE_CUDA
  switch(componentType)
    {
    case itk::ImageIOBase::CHAR:
      return ImageTypeToPixelIDValue< itk::CudaImage<int8_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::UCHAR:
      return ImageTypeToPixelIDValue< itk::CudaImage<uint8_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::SHORT:
      return ImageTypeToPixelIDValue< itk::CudaImage<int16_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::USHORT:
      return ImageTypeToPixelIDValue< itk::CudaImage<uint16_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::INT:
      return ImageTypeToPixelIDValue< itk::CudaImage<int32_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::UINT:
      return ImageTypeToPixelIDValue< itk::CudaImage<uint32_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::LONG:
      return ImageTypeToPixelIDValue< itk::CudaImage<long, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::ULONG:
      return ImageTypeToPixelIDValue< itk::CudaImage<unsigned long, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::FLOAT:
      return ImageTypeToPixelIDValue< itk::CudaImage<float, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::DOUBLE:
      return ImageTypeToPixelIDValue< itk::CudaImage<double, UnusedDimension> >::Result;
      break;
#else
  switch(componentType)
    {
    case itk::ImageIOBase::CHAR:
      return ImageTypeToPixelIDValue< itk::Image<int8_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::UCHAR:
      return ImageTypeToPixelIDValue< itk::Image<uint8_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::SHORT:
      return ImageTypeToPixelIDValue< itk::Image<int16_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::USHORT:
      return ImageTypeToPixelIDValue< itk::Image<uint16_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::INT:
      return ImageTypeToPixelIDValue< itk::Image<int32_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::UINT:
      return ImageTypeToPixelIDValue< itk::Image<uint32_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::LONG:
      return ImageTypeToPixelIDValue< itk::Image<long, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::ULONG:
      return ImageTypeToPixelIDValue< itk::Image<unsigned long, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::FLOAT:
      return ImageTypeToPixelIDValue< itk::Image<float, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::DOUBLE:
      return ImageTypeToPixelIDValue< itk::Image<double, UnusedDimension> >::Result;
      break;
#endif
    case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
    default:
      assert( false ); // should never get here unless we forgot a type
      srtkExceptionMacro( "Logic error!" );
    }
}


PixelIDValueType
ImageReaderBase
::ExecuteInternalReadComplex( int componentType )
{
  const unsigned int UnusedDimension = 2;

  switch(componentType)
    {
#ifdef RTK_USE_CUDA
    case itk::ImageIOBase::FLOAT:
      return ImageTypeToPixelIDValue< itk::CudaImage<std::complex<float>, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::DOUBLE:
      return ImageTypeToPixelIDValue< itk::CudaImage<std::complex<double>, UnusedDimension> >::Result;
      break;
#else
    case itk::ImageIOBase::FLOAT:
      return ImageTypeToPixelIDValue< itk::Image<std::complex<float>, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::DOUBLE:
      return ImageTypeToPixelIDValue< itk::Image<std::complex<double>, UnusedDimension> >::Result;
      break;
#endif
    case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
    default:
      srtkExceptionMacro( "Only Complex image with float and double are supported!" );
    }
}

PixelIDValueType
ImageReaderBase
::ExecuteInternalReadVector( int componentType )
{
  const unsigned int UnusedDimension = 2;

  switch(componentType)
    {
    case itk::ImageIOBase::CHAR:
      return ImageTypeToPixelIDValue< itk::VectorImage<signed char, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::UCHAR:
      return ImageTypeToPixelIDValue< itk::VectorImage<unsigned char, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::SHORT:
      return ImageTypeToPixelIDValue< itk::VectorImage<int16_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::USHORT:
      return ImageTypeToPixelIDValue< itk::VectorImage<uint16_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::INT:
      return ImageTypeToPixelIDValue< itk::VectorImage<int32_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::UINT:
      return ImageTypeToPixelIDValue< itk::VectorImage<uint32_t, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::LONG:
      return ImageTypeToPixelIDValue< itk::VectorImage<long, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::ULONG:
      return ImageTypeToPixelIDValue< itk::VectorImage<unsigned long, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::FLOAT:
      return ImageTypeToPixelIDValue< itk::VectorImage<float, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::DOUBLE:
      return ImageTypeToPixelIDValue< itk::VectorImage<double, UnusedDimension> >::Result;
      break;
    case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
    default:
      assert( false ); // should never get here unless we forgot a type
      srtkExceptionMacro( "Logic error!" );
    }
}


}
}
