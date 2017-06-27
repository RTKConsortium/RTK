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

#ifndef rtkSingularValueThresholdImageFilter_h
#define rtkSingularValueThresholdImageFilter_h

#include <itkInPlaceImageFilter.h>
#include <itkVector.h>

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  #include <itkImageRegionSplitterDirection.h>
#endif

namespace rtk
{
/** \class SingularValueThresholdImageFilter
 *
 * \brief Performs thresholding on the singular values
 *
 * The image is assumed to be of dimension N+1, and to contain
 * itk::CovariantVector of length N. The last dimension is assumed
 * to be the channel dimension (color, or time, or materials in spectral CT), of size L.
 * The input image must contain the spatial gradient of each channel.
 *
 * The filter walks the pixels of a single channel. For each of these pixels,
 * it constructs a matrix by concatenating the gradient vectors of the L channels.
 * The matrix is decomposed using SVD, its singular values are thresholded, and
 * then reconstructed. The resulting matrix is then cut back into L gradient vectors,
 * which are written in output.
 *
 * \author Cyril Mory
 *
 */
template< typename TInputImage,
          typename TRealType = float,
          typename TOutputImage = TInputImage>
class ITK_EXPORT SingularValueThresholdImageFilter:
        public itk::InPlaceImageFilter< TInputImage, TOutputImage >
{
public:
  
  /** Standard class typedefs. */
  typedef SingularValueThresholdImageFilter                    Self;
  typedef itk::ImageToImageFilter< TInputImage, TOutputImage > Superclass;
  typedef itk::SmartPointer< Self >                            Pointer;
  typedef itk::SmartPointer< const Self >                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkTypeMacro(SingularValueThresholdImageFilter, ImageToImageFilter);

  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  typedef typename TOutputImage::PixelType OutputPixelType;
  typedef typename TInputImage::PixelType  InputPixelType;

  /** Image typedef support */
  typedef TInputImage                       InputImageType;
  typedef TOutputImage                      OutputImageType;
  typedef typename InputImageType::Pointer  InputImagePointer;
  typedef typename OutputImageType::Pointer OutputImagePointer;

  /** The dimensionality of the input and output images. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  /** Length of the vector pixel type of the input image. */
  itkStaticConstMacro(VectorDimension, unsigned int,
                      InputPixelType::Dimension);

  /** Define the data type and the vector of data type used in calculations. */
  typedef TRealType                                                  RealType;
  typedef itk::Vector< TRealType, InputPixelType::Dimension >        RealVectorType;
  typedef itk::Image< RealVectorType, TInputImage::ImageDimension >  RealVectorImageType;

  /** Superclass typedefs. */
  typedef typename Superclass::OutputImageRegionType OutputImageRegionType;

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro( InputHasNumericTraitsCheck,
                   ( itk::Concept::HasNumericTraits< typename InputPixelType::ValueType > ) );
  itkConceptMacro( RealTypeHasNumericTraitsCheck,
                   ( itk::Concept::HasNumericTraits< RealType > ) );
  /** End concept checking */
#endif

  itkGetMacro(Threshold, TRealType)
  itkSetMacro(Threshold, TRealType)
    
protected:
  SingularValueThresholdImageFilter();
  ~SingularValueThresholdImageFilter() {}

  void BeforeThreadedGenerateData() ITK_OVERRIDE;
  void ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread,
                            itk::ThreadIdType threadId) ITK_OVERRIDE;

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
    /** Splits the OutputRequestedRegion along the first direction, not the last */
    const itk::ImageRegionSplitterBase* GetImageRegionSplitter(void) const ITK_OVERRIDE;
    itk::ImageRegionSplitterDirection::Pointer  m_Splitter;
#endif

private:
  TRealType m_Threshold;

  SingularValueThresholdImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);                     //purposely not implemented
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSingularValueThresholdImageFilter.hxx"
#endif

#endif
