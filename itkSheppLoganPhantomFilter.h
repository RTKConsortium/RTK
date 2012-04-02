#ifndef __itkSheppLoganPhantomFilter_h
#define __itkSheppLoganPhantomFilter_h

#include <itkInPlaceImageFilter.h>
#include "itkThreeDCircularProjectionGeometry.h"
#include "itkRayQuadricIntersectionImageFilter.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkRayEllipsoidIntersectionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <vector>

namespace itk
{

/** \class SheppLoganPhantomFilter
 * \brief Computes intersection of projection rays with ellipsoids.
 * in order to create a Shepp-Logan phantom projections.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT SheppLoganPhantomFilter :
  public RayEllipsoidIntersectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef SheppLoganPhantomFilter                                       Self;
  typedef RayEllipsoidIntersectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef SmartPointer<Self>                                            Pointer;
  typedef SmartPointer<const Self>                                      ConstPointer;
  typedef typename TOutputImage::RegionType               OutputImageRegionType;
  typedef typename TOutputImage::Superclass::ConstPointer OutputImageBaseConstPointer;

  typedef float OutputPixelType;

  typedef itk::Image< OutputPixelType, 3 > OutputImageType;
  typedef itk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType> REIType;
  typedef std::vector<double> VectorType;
  typedef std::string StringType;

  typedef SetQuadricParamFromRegularParamFunction                     SQPFunctionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SheppLoganPhantomFilter, RayEllipsoidIntersectionImageFilter);

  /** Get/Set Number of Figures.*/
  itkSetMacro(ConfigFile, StringType);
  itkGetMacro(ConfigFile, StringType);

  itkSetMacro(Fig, std::vector< std::vector<double> >);
  itkGetMacro(Fig, std::vector< std::vector<double> >);

protected:
  SheppLoganPhantomFilter() {}
  virtual ~SheppLoganPhantomFilter() {};

  virtual void GenerateData();
  //void Config();

  /** Translate user parameteres to quadric parameters.
   * A call to this function will assume modification of the function.*/


private:
  SheppLoganPhantomFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  std::vector< std::vector<double> > m_Fig;
  StringType m_ConfigFile;
  SQPFunctionType::Pointer m_SQPFunctor;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSheppLoganPhantomFilter.txx"
#endif

#endif
