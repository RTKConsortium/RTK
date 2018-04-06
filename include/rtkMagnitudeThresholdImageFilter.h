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

#ifndef rtkMagnitudeThresholdImageFilter_h
#define rtkMagnitudeThresholdImageFilter_h

#include <itkInPlaceImageFilter.h>
#include <itkVector.h>

namespace rtk
{
/** \class MagnitudeThresholdImageFilter
 *
 * \brief Performs thresholding on the norm of each vector-valued input pixel
 *
 * If the norm of a vector is higher than the threshold, divides the
 * components of the vector by norm / threshold. Mathematically, it amounts
 * to projecting onto the L_2 ball of radius m_Threshold
 *
 */
template< typename TInputImage,
          typename TRealType = float,
          typename TOutputImage = TInputImage>
class ITK_EXPORT MagnitudeThresholdImageFilter:
        public itk::InPlaceImageFilter< TInputImage, TOutputImage >
{
public:
  
  /** Standard class typedefs. */
  typedef MagnitudeThresholdImageFilter                        Self;
  typedef itk::ImageToImageFilter< TInputImage, TOutputImage > Superclass;
  typedef itk::SmartPointer< Self >                            Pointer;
  typedef itk::SmartPointer< const Self >                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkTypeMacro(MagnitudeThresholdImageFilter, ImageToImageFilter);

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
  MagnitudeThresholdImageFilter();
  ~MagnitudeThresholdImageFilter() {}

  void ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread,
                            itk::ThreadIdType threadId) ITK_OVERRIDE;

private:
  TRealType m_Threshold;

  MagnitudeThresholdImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);                     //purposely not implemented
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkMagnitudeThresholdImageFilter.hxx"
#endif

#endif
