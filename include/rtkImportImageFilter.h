/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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
 *
 * \ingroup RTK
 **/

template <typename TImage>
class ITK_TEMPLATE_EXPORT ImportImageFilter : public itk::ImageSource<TImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImportImageFilter);

  /** Typedef for the output image.   */
  using OutputImagePointer = typename TImage::Pointer;
  using SpacingType = typename TImage::SpacingType;
  using OriginType = typename TImage::PointType;

  /** Standard class type alias. */
  using Self = ImportImageFilter;
  using Superclass = itk::ImageSource<TImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ImportImageFilter);

  /** Index type alias support. An index is used to access pixel values. */
  using IndexType = itk::Index<TImage::ImageDimension>;

  /** Size type alias support. A size is used to define region bounds. */
  using SizeType = itk::Size<TImage::ImageDimension>;
  using SizeValueType = typename SizeType::SizeValueType;

  /** Region type alias support. A region is used to specify a
   * subset of an image. */
  using RegionType = itk::ImageRegion<TImage::ImageDimension>;

  /** Type of the output image pixel type. */
  using PixelType = typename TImage::PixelType;

  /** Get the pointer from which the image data is imported. */
  PixelType *
  GetImportPointer();

  /** Set the pointer from which the image data is imported.  "num" is
   * the number of pixels in the block of memory. If
   * "LetFilterManageMemory" is false, then the this filter will
   * not free the memory in its destructor and the application providing the
   * buffer retains the responsibility of freeing the memory for this image
   * data.  If "LetFilterManageMemory" is true, then this class
   * will free the memory when this object is destroyed. */
  void
  SetImportPointer(PixelType * ptr, SizeValueType num, bool LetFilterManageMemory);

  /** Set the region object that defines the size and starting index
   * for the imported image. This will serve as the LargestPossibleRegion,
   * the BufferedRegion, and the RequestedRegion.
   * \sa ImageRegion */
  void
  SetRegion(const RegionType & region)
  {
    if (m_Region != region)
    {
      m_Region = region;
      this->Modified();
    }
  }

  /** Get the region object that defines the size and starting index
   * for the imported image. This will serve as the LargestPossibleRegion,
   * the BufferedRegion, and the RequestedRegion.
   * \sa ImageRegion */
  const RegionType &
  GetRegion() const
  {
    return m_Region;
  }

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

  using DirectionType = itk::Matrix<double, TImage::ImageDimension, TImage::ImageDimension>;

  /** Set the direction of the image
   * \sa GetDirection() */
  virtual void
  SetDirection(const DirectionType & direction);

  /**  Get the direction of the image
   * \sa SetDirection */
  itkGetConstReferenceMacro(Direction, DirectionType);

protected:
  ImportImageFilter();
  ~ImportImageFilter() override;
  void
  PrintSelf(std::ostream & os, itk::Indent indent) const override;

  /** This filter does not actually "produce" any data, rather it "wraps"
   * the user supplied data into an itk::Image.  */
  void
  GenerateData() override;

  /** This is a source, so it must set the spacing, size, and largest possible
   * region for the output image that it will produce.
   * \sa ProcessObject::GenerateOutputInformation() */
  void
  GenerateOutputInformation() override;

  /** This filter can only produce the amount of data that it is given,
   * so we must override ProcessObject::EnlargeOutputRequestedRegion()
   * (The default implementation of a source produces the amount of
   * data requested.  This source, however, can only produce what it is
   * given.)
   *
   * \sa ProcessObject::EnlargeOutputRequestedRegion() */
  void
  EnlargeOutputRequestedRegion(itk::DataObject * output) override;

private:
  RegionType    m_Region;
  SpacingType   m_Spacing;
  OriginType    m_Origin;
  DirectionType m_Direction;

  PixelType *   m_ImportPointer;
  bool          m_FilterManageMemory;
  SizeValueType m_Size;
};
} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkImportImageFilter.hxx"
#endif

#endif
