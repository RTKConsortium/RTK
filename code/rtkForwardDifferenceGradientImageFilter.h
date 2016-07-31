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
#ifndef rtkForwardDifferenceGradientImageFilter_h
#define rtkForwardDifferenceGradientImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkCovariantVector.h>
#include <itkImageRegionIterator.h>

#include "rtkMacro.h"
#include <itkVectorImage.h>

namespace rtk
{

//template <typename TPixelType, unsigned int VImageDimension > class itk::VectorImage;


/** \class ForwardDifferenceGradientImageFilter
 * \brief Computes the gradient of an image using forward difference.
 *
 * Computes the gradient of an image using directional forward differences.
 * The directional derivative at each pixel location is computed by
 * convolution with a first-order derivative operator.
 *
 * The second template parameter defines the value type used in the
 * derivative operator (defaults to float).  The third template
 * parameter defines the value type used for output image (defaults to
 * float).  The output image is defined as a covariant vector image
 * whose value type is specified as this third template parameter.
 *
 * The exact definition of the desired gradient filter can
 * be found in Chambolle, Antonin. “An Algorithm for Total
 * Variation Minimization and Applications.” J. Math. Imaging Vis. 20,
 * no. 1–2 (January 2004): 89–97.
 *
 * \sa Image
 * \sa Neighborhood
 * \sa NeighborhoodOperator
 * \sa NeighborhoodIterator
 *
 * \ingroup GradientFilters
 * \ingroup ITKImageGradient
 */
template< typename TInputImage,
          typename TOperatorValueType = float,
          typename TOutputValueType = float,
          typename TOutputImageType = itk::Image< itk::CovariantVector< TOutputValueType,
                                                           TInputImage::ImageDimension >,
                                          TInputImage::ImageDimension > >
class ForwardDifferenceGradientImageFilter:
  public itk::ImageToImageFilter< TInputImage, TOutputImageType >
{
public:
  /** Extract dimension from input image. */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImageType::ImageDimension);

  /** Convenient typedefs for simplifying declarations. */
  typedef TInputImage                       InputImageType;
  typedef typename InputImageType::Pointer  InputImagePointer;
  typedef TOutputImageType                  OutputImageType;
  typedef typename OutputImageType::Pointer OutputImagePointer;

  /** Standard class typedefs. */
  typedef ForwardDifferenceGradientImageFilter                       Self;
  typedef itk::ImageToImageFilter< InputImageType, OutputImageType > Superclass;
  typedef itk::SmartPointer< Self >                                  Pointer;
  typedef itk::SmartPointer< const Self >                            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ForwardDifferenceGradientImageFilter, ImageToImageFilter);

  /** Image typedef support. */
  typedef typename InputImageType::PixelType  InputPixelType;
  typedef TOperatorValueType                  OperatorValueType;
  typedef TOutputValueType                    OutputValueType;

  typedef typename OutputImageType::PixelType OutputPixelType;

  typedef itk::CovariantVector<
    OutputValueType, itkGetStaticConstMacro(OutputImageDimension) >
                                              CovariantVectorType;

  typedef typename OutputImageType::RegionType OutputImageRegionType;

  /** ForwardDifferenceGradientImageFilter needs a larger input requested region than
   * the output requested region.  As such, ForwardDifferenceGradientImageFilter needs
   * to provide an implementation for GenerateInputRequestedRegion()
   * in order to inform the pipeline execution model.
   *
   * \sa ImageToImageFilter::GenerateInputRequestedRegion() */
  virtual void GenerateInputRequestedRegion() ITK_OVERRIDE;

  /** Use the image spacing information in calculations. Use this option if you
   *  want derivatives in physical space. Default is UseImageSpacingOn. */
  void SetUseImageSpacingOn()
  { this->SetUseImageSpacing(true); }

  /** Ignore the image spacing. Use this option if you want derivatives in
      isotropic pixel space.  Default is UseImageSpacingOn. */
  void SetUseImageSpacingOff()
  { this->SetUseImageSpacing(false); }

  /** Set/Get whether or not the filter will use the spacing of the input
      image in its calculations */
  itkSetMacro(UseImageSpacing, bool);
  itkGetConstMacro(UseImageSpacing, bool);
  itkBooleanMacro(UseImageSpacing);

  /** Set along which dimensions the gradient computation should be
      performed. The vector components at unprocessed dimensions contain
      undetermined values */
  void SetDimensionsProcessed(bool* DimensionsProcessed);

  /** Allows to change the default boundary condition */
  void OverrideBoundaryCondition(itk::ImageBoundaryCondition< TInputImage >* boundaryCondition);

#ifdef ITK_USE_CONCEPT_CHECKING
  // Begin concept checking
  itkConceptMacro( InputConvertibleToOutputCheck,
                   ( itk::Concept::Convertible< InputPixelType, OutputValueType > ) );
  itkConceptMacro( OutputHasNumericTraitsCheck,
                   ( itk::Concept::HasNumericTraits< OutputValueType > ) );
  // End concept checking
#endif

  /** The UseImageDirection flag determines whether image derivatives are
   * computed with respect to the image grid or with respect to the physical
   * space. When this flag is ON the derivatives are computed with respect to
   * the coodinate system of physical space. The difference is whether we take
   * into account the image Direction or not. The flag ON will take into
   * account the image direction and will result in an extra matrix
   * multiplication compared to the amount of computation performed when the
   * flag is OFF.
   * The default value of this flag is On.
   */
  itkSetMacro(UseImageDirection, bool);
  itkGetConstMacro(UseImageDirection, bool);
  itkBooleanMacro(UseImageDirection);

protected:
  ForwardDifferenceGradientImageFilter();
  virtual ~ForwardDifferenceGradientImageFilter() ITK_OVERRIDE;
  void PrintSelf(std::ostream & os, itk::Indent indent) const ITK_OVERRIDE;

  /** ForwardDifferenceGradientImageFilter can be implemented as a multithreaded filter.
   * Therefore, this implementation provides a ThreadedGenerateData()
   * routine which is called for each processing thread. The output
   * image data is allocated automatically by the superclass prior to
   * calling ThreadedGenerateData().  ThreadedGenerateData can only
   * write to the portion of the output image specified by the
   * parameter "outputRegionForThread"
   *
   * \sa ImageToImageFilter::ThreadedGenerateData(),
   *     ImageToImageFilter::GenerateData() */
  void ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread,
                            itk::ThreadIdType threadId) ITK_OVERRIDE;

private:
#ifdef ITK_DISALLOW_COPY_AND_ASSIGN
  ITK_DISALLOW_COPY_AND_ASSIGN(ForwardDifferenceGradientImageFilter);
#else
  ForwardDifferenceGradientImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);      //purposely not implemented
#endif

  virtual void GenerateOutputInformation() ITK_OVERRIDE;

  // An overloaded method which may transform the gradient to a
  // physical vector and converts to the correct output pixel type.
  template <typename TValue>
  void SetOutputPixel( itk::ImageRegionIterator< itk::VectorImage<TValue,OutputImageDimension> > &it, CovariantVectorType &gradient )
  {
    if ( this->m_UseImageDirection )
      {
      CovariantVectorType physicalGradient;
      it.GetImage()->TransformLocalVectorToPhysicalVector( gradient, physicalGradient );
      it.Set( OutputPixelType( physicalGradient.GetDataPointer(), InputImageDimension, false ) );
      }
    else
      {
      it.Set( OutputPixelType( gradient.GetDataPointer(), InputImageDimension, false ) );
      }
  }

  template <typename T >
  void SetOutputPixel( itk::ImageRegionIterator< T > &it, CovariantVectorType &gradient )
  {
    // This uses the more efficient set by reference method
    if ( this->m_UseImageDirection )
      {
      CovariantVectorType physicalGradient;
      for( unsigned int i =0; i <  OutputPixelType::Dimension ; ++i )
        {
        physicalGradient[i] = it.Value()[i];
        }
      it.GetImage()->TransformLocalVectorToPhysicalVector( gradient, physicalGradient);
      for( unsigned int i =0; i <  OutputPixelType::Dimension ; ++i )
        {
        it.Value()[i] = physicalGradient[i];
        }
      }
    else
      {
      for( unsigned int i =0; i <  OutputPixelType::Dimension ; ++i )
        {
        it.Value()[i] = gradient[i];
        }
      }
  }


  bool m_UseImageSpacing;

  // flag to take or not the image direction into account
  // when computing the derivatives.
  bool m_UseImageDirection;

  // list of the dimensions along which the gradient has
  // to be computed. The components on other dimensions
  // are set to zero
  bool m_DimensionsProcessed[TInputImage::ImageDimension];

  itk::ImageBoundaryCondition< TInputImage, TInputImage >* m_BoundaryCondition;
  bool                                                     m_IsBoundaryConditionOverriden;
};
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkForwardDifferenceGradientImageFilter.hxx"
#endif

#endif
