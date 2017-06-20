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
#ifndef __srtkImportImageFilter_h
#define __srtkImportImageFilter_h

#include "srtkMacro.h"
#include "srtkImage.h"
#include "srtkImageReaderBase.h"
#include "srtkMemberFunctionFactory.h"

namespace rtk {
  namespace simple {

    /** \class ImportImageFilter
     * \brief Compose a 2D or 3D image and return a smart pointer to a SimpleRTK
     * image
     *
     * This filter is intended to interface SimpleRTK to other image processing
     * libraries and applications that may have their own representation of an
     * image class. It creates a SimpleITK image which shares the bulk
     * data buffer as what is set. SimpleITK will not responsible to
     * delete the buffer afterwards, and it buffer must remain valid
     * while in use.
     *
     * \sa rtk::simple::ImportAsInt8, rtk::simple::ImportAsUInt8,
     * rtk::simple::ImportAsInt16, rtk::simple::ImportAsUInt16,
     * rtk::simple::ImportAsInt32, rtk::simple::ImportAsUInt32,
     * rtk::simple::ImportAsInt64, rtk::simple::ImportAsUInt64,
     * rtk::simple::ImportAsFloat,rtk::simple::ImportAsDouble for the
     * procedural interfaces.
     */
    class SRTKIO_EXPORT ImportImageFilter
      : public ImageReaderBase {
    public:
      typedef ImportImageFilter Self;

      ImportImageFilter();

      /** Print ourselves to string */
      virtual std::string ToString() const;

      /** return user readable name fo the filter */
      virtual std::string GetName() const { return std::string("ImportImageFilter"); }

      SRTK_RETURN_SELF_TYPE_HEADER SetSize( const std::vector< unsigned int > &size );
      const std::vector< unsigned int > &GetSize( ) const;

      SRTK_RETURN_SELF_TYPE_HEADER SetSpacing( const std::vector< double > &spacing );
      const std::vector< double > &GetSpacing( ) const;

      SRTK_RETURN_SELF_TYPE_HEADER SetOrigin( const std::vector< double > &origin );
      const std::vector< double > &GetOrigin( ) const;

      SRTK_RETURN_SELF_TYPE_HEADER SetDirection( const std::vector< double > &direction );
      const std::vector< double > &GetDirection( ) const;

      SRTK_RETURN_SELF_TYPE_HEADER SetBufferAsInt8( int8_t * buffer, unsigned int numberOfComponents = 1 );
      SRTK_RETURN_SELF_TYPE_HEADER SetBufferAsUInt8( uint8_t * buffer, unsigned int numberOfComponents = 1 );
      SRTK_RETURN_SELF_TYPE_HEADER SetBufferAsInt16( int16_t * buffer, unsigned int numberOfComponents = 1 );
      SRTK_RETURN_SELF_TYPE_HEADER SetBufferAsUInt16( uint16_t * buffer, unsigned int numberOfComponents = 1 );
      SRTK_RETURN_SELF_TYPE_HEADER SetBufferAsInt32( int32_t * buffer, unsigned int numberOfComponents = 1 );
      SRTK_RETURN_SELF_TYPE_HEADER SetBufferAsUInt32( uint32_t * buffer, unsigned int numberOfComponents = 1 );
      SRTK_RETURN_SELF_TYPE_HEADER SetBufferAsInt64( int64_t * buffer, unsigned int numberOfComponents = 1 );
      SRTK_RETURN_SELF_TYPE_HEADER SetBufferAsUInt64( uint64_t * buffer, unsigned int numberOfComponents = 1 );
      SRTK_RETURN_SELF_TYPE_HEADER SetBufferAsFloat( float * buffer, unsigned int numberOfComponents = 1 );
      SRTK_RETURN_SELF_TYPE_HEADER SetBufferAsDouble( double * buffer, unsigned int numberOfComponents = 1 );

      Image Execute();

    protected:

      // Internal method called the the template dispatch system
      template <class TImageType> Image ExecuteInternal ( void );

      // If the output image type is a VectorImage then the number of
      // components per pixel needs to be set, otherwise the method
      // does not exist. This is done with the EnableIf Idiom.
      template <class TImageType>
      typename DisableIf<IsVector<TImageType>::Value>::Type
      SetNumberOfComponentsOnImage( TImageType* ) {}
      template <class TImageType>
      typename EnableIf<IsVector<TImageType>::Value>::Type
      SetNumberOfComponentsOnImage( TImageType* );

    private:

      // function pointer type
      typedef Image (Self::*MemberFunctionType)( void );

      // friend to get access to executeInternal member
      friend struct detail::MemberFunctionAddressor<MemberFunctionType>;
      nsstd::auto_ptr<detail::MemberFunctionFactory<MemberFunctionType> > m_MemberFactory;

      unsigned int     m_NumberOfComponentsPerPixel;
      PixelIDValueType m_PixelIDValue;

      std::vector< double >         m_Origin;
      std::vector< double >         m_Spacing;
      std::vector< unsigned int >   m_Size;
      std::vector< double >         m_Direction;

      void        * m_Buffer;

    };

  Image SRTKIO_EXPORT ImportAsInt8(
    int8_t * buffer,
    const std::vector< unsigned int > &size,
    const std::vector< double > &spacing = std::vector< double >( 3, 1.0 ),
    const std::vector< double > &origin = std::vector< double >( 3, 0.0 ),
    const std::vector< double > &direction = std::vector< double >(),
    unsigned int numberOfComponents = 1
    );

  Image SRTKIO_EXPORT ImportAsUInt8(
    uint8_t * buffer,
    const std::vector< unsigned int > &size,
    const std::vector< double > &spacing = std::vector< double >( 3, 1.0 ),
    const std::vector< double > &origin = std::vector< double >( 3, 0.0 ),
    const std::vector< double > &direction = std::vector< double >(),
    unsigned int numberOfComponents = 1
    );

  Image SRTKIO_EXPORT ImportAsInt16(
    int16_t * buffer,
    const std::vector< unsigned int > &size,
    const std::vector< double > &spacing = std::vector< double >( 3, 1.0 ),
    const std::vector< double > &origin = std::vector< double >( 3, 0.0 ),
    const std::vector< double > &direction = std::vector< double >(),
    unsigned int numberOfComponents = 1
    );

  Image SRTKIO_EXPORT ImportAsUInt16(
    uint16_t * buffer,
    const std::vector< unsigned int > &size,
    const std::vector< double > &spacing = std::vector< double >( 3, 1.0 ),
    const std::vector< double > &origin = std::vector< double >( 3, 0.0 ),
    const std::vector< double > &direction = std::vector< double >(),
    unsigned int numberOfComponents = 1
    );

  Image SRTKIO_EXPORT ImportAsInt32(
    int32_t * buffer,
    const std::vector< unsigned int > &size,
    const std::vector< double > &spacing = std::vector< double >( 3, 1.0 ),
    const std::vector< double > &origin = std::vector< double >( 3, 0.0 ),
    const std::vector< double > &direction = std::vector< double >(),
    unsigned int numberOfComponents = 1
    );

  Image SRTKIO_EXPORT ImportAsUInt32(
    uint32_t * buffer,
    const std::vector< unsigned int > &size,
    const std::vector< double > &spacing = std::vector< double >( 3, 1.0 ),
    const std::vector< double > &origin = std::vector< double >( 3, 0.0 ),
    const std::vector< double > &direction = std::vector< double >(),
    unsigned int numberOfComponents = 1
    );

  Image SRTKIO_EXPORT ImportAsInt64(
    int64_t * buffer,
    const std::vector< unsigned int > &size,
    const std::vector< double > &spacing = std::vector< double >( 3, 1.0 ),
    const std::vector< double > &origin = std::vector< double >( 3, 0.0 ),
    const std::vector< double > &direction = std::vector< double >(),
    unsigned int numberOfComponents = 1
    );

  Image SRTKIO_EXPORT ImportAsUInt64(
    uint64_t * buffer,
    const std::vector< unsigned int > &size,
    const std::vector< double > &spacing = std::vector< double >( 3, 1.0 ),
    const std::vector< double > &origin = std::vector< double >( 3, 0.0 ),
    const std::vector< double > &direction = std::vector< double >(),
    unsigned int numberOfComponents = 1
    );

  Image SRTKIO_EXPORT ImportAsFloat(
    float * buffer,
    const std::vector< unsigned int > &size,
    const std::vector< double > &spacing = std::vector< double >( 3, 1.0 ),
    const std::vector< double > &origin = std::vector< double >( 3, 0.0 ),
    const std::vector< double > &direction = std::vector< double >(),
    unsigned int numberOfComponents = 1
    );

  Image SRTKIO_EXPORT ImportAsDouble(
    double * buffer,
    const std::vector< unsigned int > &size,
    const std::vector< double > &spacing = std::vector< double >( 3, 1.0 ),
    const std::vector< double > &origin = std::vector< double >( 3, 0.0 ),
    const std::vector< double > &direction = std::vector< double >(),
    unsigned int numberOfComponents = 1
    );

  }
}

#endif
