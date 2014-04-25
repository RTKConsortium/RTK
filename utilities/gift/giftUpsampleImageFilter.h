/*=========================================================================

  Program:  GIFT (Generalised Image Fusion Toolkit)
  Module:   giftUpsampleImageFilter.h
  Language: C++
  Date:     2005/11/16
  Version:  1.0
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftUpsampleImageFilter_H
#define __giftUpsampleImageFilter_H

#include "itkImageToImageFilter.h"

namespace gift
{

/** \class UpsampleImageFilter
 * \brief Upsamples an image by the given factor for each dimension.
 *
 * Since this filter produces an image which is a different resolution 
 * and with different pixel spacing than its input image, 
 * it needs to override several of the methods defined
 * in ProcessObject in order to properly manage the pipeline execution model.
 * In particular, this filter overrides
 * ProcessObject::GenerateInputRequestedRegion() and
 * ProcessObject::GenerateOutputInformation().
 *
 * This filter is implemented as a multithreaded filter.  It provides a 
 * ThreadedGenerateData() method for its implementation.
 * 
 * \ingroup GeometricTransforms
 */
template <class TInputImage, class TOutputImage = TInputImage>
class ITK_EXPORT UpsampleImageFilter:
    public itk::ImageToImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef UpsampleImageFilter                                   Self;
  typedef itk::ImageToImageFilter<TInputImage,TOutputImage>     Superclass;
  typedef itk::SmartPointer<Self>                               Pointer;
  typedef itk::SmartPointer<const Self>                         ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);  

  /** Run-time type information (and related methods). */
  itkTypeMacro(UpsampleImageFilter, ImageToImageFilter);

  /** Typedef to images */
  typedef TOutputImage                                OutputImageType;
  typedef TInputImage                                 InputImageType;
  typedef typename OutputImageType::Pointer           OutputImagePointer;
  typedef typename InputImageType::Pointer            InputImagePointer;
  typedef typename InputImageType::ConstPointer       InputImageConstPointer;

  /** Typedef to describe the output image region type. */
  typedef typename TOutputImage::RegionType OutputImageRegionType;

  /** ImageDimension enumeration. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension );

  /** Set the shrink factors. Values are clamped to
   * a minimum value of 1.*/
  void SetFactors(unsigned int factors[]);
  
  /** Sets the shrink factor for the given dimension.
   * All other dimensions are set to 1 */
  void SetFactor(unsigned int dimension, unsigned int factor);

  /** UpsampleImageFilter produces an image which is a different
   * resolution and with a different pixel spacing than its input
   * image.  As such, UpsampleImageFilter needs to provide an
   * implementation for GenerateOutputInformation() in order to inform
   * the pipeline execution model.  The original documentation of this
   * method is below.
   * \sa ProcessObject::GenerateOutputInformaton() */
  virtual void GenerateOutputInformation();

  /** UpsampleImageFilter needs a larger input requested region than the output
   * requested region.  As such, UpsampleImageFilter needs to provide an
   * implementation for GenerateInputRequestedRegion() in order to inform the
   * pipeline execution model.
   * \sa ProcessObject::GenerateInputRequestedRegion() */
  virtual void GenerateInputRequestedRegion();

  /** Set/Get the order of the wavelet filter
   * This is required because some information about the index of the image
   * is lost during downsampling, and the upsampling filter can't guess
   * what the exact index should be.
   */
  itkSetMacro(Order, unsigned int)
  itkGetMacro(Order, unsigned int)

protected:
  UpsampleImageFilter();
  ~UpsampleImageFilter() {};

  /** UpsampleImageFilter can be implemented as a multithreaded filter.
   * Therefore, this implementation provides a ThreadedGenerateData() routine
   * which is called for each processing thread. The output image data is
   * allocated automatically by the superclass prior to calling
   * ThreadedGenerateData().  ThreadedGenerateData can only write to the
   * portion of the output image specified by the parameter
   * "outputRegionForThread"
   *
   * \sa ImageToImageFilter::ThreadedGenerateData(),
   *     ImageToImageFilter::GenerateData() */
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId));

private:
  UpsampleImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);          //purposely not implemented

  unsigned int m_Factors[ImageDimension];
  unsigned int m_Order;
};

  
} // end namespace gift
  
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftUpsampleImageFilter.txx"
#endif
  
#endif
