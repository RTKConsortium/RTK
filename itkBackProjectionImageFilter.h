#ifndef __itkBackProjectionImageFilter_h
#define __itkBackProjectionImageFilter_h

#include "itkInPlaceImageFilter.h"
#include "itkConceptChecking.h"
#include "itkProjectionGeometry.h"

namespace itk
{

template <class TInputImage, class TOutputImage>
class ITK_EXPORT BackProjectionImageFilter :
  public InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef BackProjectionImageFilter                    Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage> Superclass;
  typedef SmartPointer<Self>                           Pointer;
  typedef SmartPointer<const Self>                     ConstPointer;
  typedef typename TInputImage::PixelType              InputPixelType;
  typedef typename TOutputImage::RegionType            OutputImageRegionType;

  typedef itk::ProjectionGeometry<TOutputImage::ImageDimension>     GeometryType;
  typedef typename GeometryType::Pointer                            GeometryPointer;
  typedef typename GeometryType::MatrixType                         ProjectionMatrixType;
  typedef itk::Image<InputPixelType, TInputImage::ImageDimension-1> ProjectionImageType;
  typedef typename ProjectionImageType::Pointer                     ProjectionImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BackProjectionImageFilter, ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

  /** Get / Set the transpose flag for 2D projections (optimization trick) */
  itkGetMacro(Transpose, bool);
  itkSetMacro(Transpose, bool);

  /** Get / Set the flag to update one projection at a time during backprojection
   * instead of requiring for the LargestPossibleRegion. This can decrease the
   * amount of memory required but increases the computation time when using multithreading.
   */
  itkGetMacro(UpdateProjectionPerProjection, bool);
  itkSetMacro(UpdateProjectionPerProjection, bool);
protected:
  BackProjectionImageFilter() : m_Geometry(NULL), m_Transpose(false) {
    this->SetNumberOfRequiredInputs(2); this->SetInPlace( true ); m_ProjectionStackLock = FastMutexLock::New();
  };
  virtual ~BackProjectionImageFilter() {
  };

  /** Apply changes to the input image requested region. */
  virtual void GenerateInputRequestedRegion();

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, int threadId );

  /** The input is a stack of projections, we need to interpolate in one projection
      for efficiency during interpolation. Use of itk::ExtractImageFilter is
      not threadsafe in ThreadedGenerateData, this one is. The output can be multiplied by a constant. */
  ProjectionImagePointer GetProjection(const unsigned int iProj);

  /** Creates the #iProj index to index projection matrix with current inputs
      instead of the physical point to physical point projection matrix provided by Geometry */
  ProjectionMatrixType GetIndexToIndexProjectionMatrix(const unsigned int iProj, const ProjectionImageType *proj);

private:
  BackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  /** RTK geometry object */
  GeometryPointer m_Geometry;

  /** Flip projection flag: infludences GetProjection and
    GetIndexToIndexProjectionMatrix for optimization */
  bool m_Transpose;

  /** Update the requested projection in GetProjection instead of asking for the LargestPossibleRegion in
   * GenerateInputRequestedRegion. */
  bool                   m_UpdateProjectionPerProjection;
  FastMutexLock::Pointer m_ProjectionStackLock;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBackProjectionImageFilter.txx"
#endif

#endif
