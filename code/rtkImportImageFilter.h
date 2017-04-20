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

#ifndef rtkImportImageFilter_h
#define rtkImportImageFilter_h

#include "itkImageSource.h"
#include "rtkMacro.h"

namespace rtk
{
/** \class ImportImageFilter
 * \brief Import data from a standard C array into an itk::Image
 *
 * ImportImageFilter provides a mechanism for importing data into a TImage.
 * ImportImageFilter is an image source, so it behaves like any other pipeline
 * object.
 *
 * This class is templated over the image type of the output image, unlike
 * itk::ImportImageFilter which is templated over the pixel type and the dimension
 * and is therefore incompatible with itk::CudaImage.
 *
 * \author Marc Vila
 **/

template< typename TImage >
class ImportImageFilter : public itk::ImageSource<TImage>
{
public:
  /** Typedef for the output image.   */
  typedef typename TImage::Pointer        OutputImagePointer;
  typedef typename TImage::SpacingType    SpacingType;
  typedef typename TImage::PointType      OriginType;

  /** Standard class typedefs. */
  typedef ImportImageFilter                Self;
  typedef itk::ImageSource<TImage>         Superclass;
  typedef itk::SmartPointer< Self >        Pointer;
  typedef itk::SmartPointer< const Self >  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImportImageFilter, ImageSource);

  /** Index typedef support. An index is used to access pixel values. */
  typedef itk::Index< TImage::ImageDimension > IndexType;

  /** Size typedef support. A size is used to define region bounds. */
  typedef itk::Size< TImage::ImageDimension > SizeType;
  typedef typename SizeType::SizeValueType    SizeValueType;

  /** Region typedef support. A region is used to specify a
   * subset of an image. */
  typedef itk::ImageRegion< TImage::ImageDimension > RegionType;

  /** Type of the output image pixel type. */
  typedef typename TImage::PixelType PixelType;

  /** Get the pointer from which the image data is imported. */
  PixelType * GetImportPointer();

  /** Set the pointer from which the image data is imported.  "num" is
   * the number of pixels in the block of memory. If
   * "LetFilterManageMemory" is false, then the this filter will
   * not free the memory in its destructor and the application providing the
   * buffer retains the responsibility of freeing the memory for this image
   * data.  If "LetFilterManageMemory" is true, then this class
   * will free the memory when this object is destroyed. */
  void SetImportPointer(PixelType  *ptr, SizeValueType num,
                        bool LetFilterManageMemory);

  /** Set the region object that defines the size and starting index
   * for the imported image. This will serve as the LargestPossibleRegion,
   * the BufferedRegion, and the RequestedRegion.
   * \sa ImageRegion */
  void SetRegion(const RegionType & region)
  { if ( m_Region != region ) { m_Region = region; this->Modified(); } }

  /** Get the region object that defines the size and starting index
   * for the imported image. This will serve as the LargestPossibleRegion,
   * the BufferedRegion, and the RequestedRegion.
   * \sa ImageRegion */
  const RegionType & GetRegion() const
  { return m_Region; }

  /** Set the spacing (size of a pixel) of the image.
   * \sa GetSpacing() */
  itkSetMacro(Spacing, SpacingType);
  itkGetConstReferenceMacro(Spacing, SpacingType);
  itkSetVectorMacro(Spacing, const float, TImage::ImageDimension);

  /** Set the origin of the image.
   * \sa GetOrigin() */
  itkSetMacro(Origin, OriginType);
  itkGetConstReferenceMacro(Origin, OriginType);
  itkSetVectorMacro(Origin, const float, TImage::ImageDimension);

  typedef itk::Matrix< double, TImage::ImageDimension, TImage::ImageDimension > DirectionType;

  /** Set the direction of the image
   * \sa GetDirection() */
  virtual void SetDirection(const DirectionType & direction);

  /**  Get the direction of the image
   * \sa SetDirection */
  itkGetConstReferenceMacro(Direction, DirectionType);

protected:
  ImportImageFilter();
  ~ImportImageFilter();
  void PrintSelf(std::ostream & os, itk::Indent indent) const ITK_OVERRIDE;

  /** This filter does not actually "produce" any data, rather it "wraps"
   * the user supplied data into an itk::Image.  */
  void GenerateData() ITK_OVERRIDE;

  /** This is a source, so it must set the spacing, size, and largest possible
   * region for the output image that it will produce.
   * \sa ProcessObject::GenerateOutputInformation() */
  void GenerateOutputInformation() ITK_OVERRIDE;

  /** This filter can only produce the amount of data that it is given,
   * so we must override ProcessObject::EnlargeOutputRequestedRegion()
   * (The default implementation of a source produces the amount of
   * data requested.  This source, however, can only produce what it is
   * given.)
   *
   * \sa ProcessObject::EnlargeOutputRequestedRegion() */
  void EnlargeOutputRequestedRegion(itk::DataObject *output) ITK_OVERRIDE;

private:
  ImportImageFilter(const ImportImageFilter &); //purposely not implemented
  void operator=(const ImportImageFilter &);    //purposely not implemented

  RegionType    m_Region;
  SpacingType   m_Spacing;
  OriginType    m_Origin;
  DirectionType m_Direction;

  PixelType  *  m_ImportPointer;
  bool          m_FilterManageMemory;
  SizeValueType m_Size;
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkImportImageFilter.hxx"
#endif

#endif
