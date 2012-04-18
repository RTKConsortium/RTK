#ifndef __itkDrawQuadricFunctor_h
#define __itkDrawQuadricFunctor_h

#include <itkInPlaceImageFilter.h>
#include "itkThreeDCircularProjectionGeometry.h"
#include "itkRayQuadricIntersectionImageFilter.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkInPlaceImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <vector>

namespace itk
{

/** \class DrawQuadricFunctor
 * \brief Computes the 3D reference of an specefic quadric surface.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT DrawQuadricFunctor :
  public InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef DrawQuadricFunctor                                        Self;
  typedef InPlaceImageFilter<TInputImage,TOutputImage>              Superclass;
  typedef SmartPointer<Self>                                        Pointer;
  typedef SmartPointer<const Self>                                  ConstPointer;
  typedef typename TOutputImage::RegionType                         OutputImageRegionType;
  typedef typename TOutputImage::Superclass::ConstPointer           OutputImageBaseConstPointer;

  typedef float OutputPixelType;

  typedef itk::Image< OutputPixelType, 3 >                          OutputImageType;
  typedef std::vector<double>                                       VectorType;
  typedef std::string                                               StringType;

  typedef SetQuadricParamFromRegularParamFunction                   SQPFunctionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawQuadricFunctor, InPlaceImageFilter);

  /** Get/Set ConfigFile*/
  itkSetMacro(ConfigFile, StringType);
  itkGetMacro(ConfigFile, StringType);

protected:
  DrawQuadricFunctor() {}
  virtual ~DrawQuadricFunctor() {};

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );
  /** Translate user parameteres to quadric parameters.
   * A call to this function will assume modification of the function.*/


private:
  DrawQuadricFunctor(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented
  SQPFunctionType::Pointer m_SQPFunctor;
  StringType m_ConfigFile;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDrawQuadricFunctor.txx"
#endif

#endif


