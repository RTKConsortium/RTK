#ifndef __itkBackProjectionImageFilter_h
#define __itkBackProjectionImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkConceptChecking.h"
#include "rtkGeometry.h"

namespace itk
{
  
template <class TInputImage, class TOutputImage>
class ITK_EXPORT BackProjectionImageFilter :
  public ImageToImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef BackProjectionImageFilter                         Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BackProjectionImageFilter, ImageToImageFilter);

  /** Some convenient typedefs. */
  typedef TInputImage                                 InputImageType;
  typedef typename    InputImageType::Pointer         InputImagePointer;
  typedef typename    InputImageType::RegionType      InputImageRegionType;
  typedef typename    InputImageType::PixelType       InputImagePixelType;
  typedef typename    InputImageType::PointType       InputImagePointType;
  typedef TOutputImage                                OutputImageType;
  typedef typename    OutputImageType::Pointer        OutputImagePointer;
  typedef typename    OutputImageType::RegionType     OutputImageRegionType;
  typedef typename    OutputImageType::PixelType      OutputImagePixelType;
  typedef typename    OutputImageType::PointType      OutputImagePointType;
  typedef typename    OutputImageType::SizeType       OutputImageSizeType;
  typedef typename    OutputImageType::SpacingType    OutputImageSpacingType;

  typedef rtk::Geometry<TOutputImage::ImageDimension> GeometryType;
  typedef typename    GeometryType::Pointer           GeometryPointer;

  /** Set the geometry containing projection geometry */
  itkSetMacro(Geometry, GeometryPointer);

  /** Set number of projections skipped after each bp */
  itkSetMacro(SkipProjection, unsigned int);

  /** Get the information of the tomography */
  itkGetConstMacro(TomographyDimension, OutputImageSizeType);
  itkGetConstMacro(TomographySpacing, OutputImageSpacingType);
  itkGetConstMacro(TomographyOrigin, OutputImagePointType);

  /** Set the information of the tomography */
  itkSetMacro(TomographyDimension, OutputImageSizeType);
  itkSetMacro(TomographySpacing, OutputImageSpacingType);
  itkSetMacro(TomographyOrigin, OutputImagePointType);

  /** Set filter options from gengetopt generated struct. The ggo file should contain:
    section "Backprojection"
    option "skip_proj" s "Skip projections"          int       no   default="0"
    option "origin"    - "Origin (default=centered)" double multiple no
    option "dimension" - "Dimension"                 int    multiple no  default="256"
    option "spacing"   - "Spacing"                   double multiple no  default="1"
   */
  template <class Args_Info> void SetFromGengetopt(const Args_Info & args_info);

protected:
  BackProjectionImageFilter():m_SkipProjection(0) {};
  virtual ~BackProjectionImageFilter() {};

  /** Apply changes to the output image information. */
  virtual void GenerateOutputInformation();

  /** Apply changes to the input image requested region. */
  virtual void GenerateInputRequestedRegion();

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, int threadId );

private:
  BackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** RTK geometry object */
  GeometryPointer m_Geometry;

  /** Number of projection to skip after backprojecting one */
  unsigned int m_SkipProjection;

  /** Output tomography information */
  OutputImageSizeType m_TomographyDimension;
  OutputImageSpacingType m_TomographySpacing;
  OutputImagePointType m_TomographyOrigin;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBackProjectionImageFilter.txx"
#endif

#endif
