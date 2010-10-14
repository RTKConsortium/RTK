#ifndef __itkBackProjectionImageFilter_h
#define __itkBackProjectionImageFilter_h

#include "itkInPlaceImageFilter.h"
#include "itkConceptChecking.h"
#include "rtkGeometry.h"

namespace itk
{
  
template <class TInputImage, class TOutputImage>
class ITK_EXPORT BackProjectionImageFilter :
  public InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef BackProjectionImageFilter                                 Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>              Superclass;
  typedef SmartPointer<Self>                                        Pointer;
  typedef SmartPointer<const Self>                                  ConstPointer;
  typedef typename TInputImage::PixelType                           InputPixelType;

  typedef rtk::Geometry<TOutputImage::ImageDimension>               GeometryType;
  typedef typename GeometryType::Pointer                            GeometryPointer;
  typedef typename GeometryType::MatrixType                         ProjectionMatrixType;
  typedef itk::Image<InputPixelType, TInputImage::ImageDimension-1> ProjectionImageType;
  typedef typename ProjectionImageType::Pointer                     ProjectionImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BackProjectionImageFilter, ImageToImageFilter);

  /** Set the geometry containing projection geometry */
  itkSetMacro(Geometry, GeometryPointer);

protected:
  BackProjectionImageFilter() {this->SetNumberOfRequiredInputs(2); this->SetInPlace( true ); };
  virtual ~BackProjectionImageFilter() {};

  /** Apply changes to the input image requested region. */
  virtual void GenerateInputRequestedRegion();

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, int threadId );

  /** The input is a stack of projections, we need to interpolate in one projection
      for efficiency during interpolation. Use of itk::ExtractImageFilter is
      not threadsafe in ThreadedGenerateData, this one is. */
  typename ProjectionImagePointer GetProjection(const unsigned int iProj);

  /** Creates the #iProj index to index projection matrix with current inputs
      instead of the physical point to physical point projection matrix provided by Geometry */
  typename ProjectionMatrixType GetIndexToIndexProjectionMatrix(const unsigned int iProj, const ProjectionImageType *proj);

private:
  BackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** RTK geometry object */
  GeometryPointer m_Geometry;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBackProjectionImageFilter.txx"
#endif

#endif
