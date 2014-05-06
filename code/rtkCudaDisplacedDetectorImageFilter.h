//
#ifndef RTKCUDADISPLACEDDETECTORIMAGEFILTER_H
#define RTKCUDADISPLACEDDETECTORIMAGEFILTER_H

#include "rtkDisplacedDetectorImageFilter.h"
#include "rtkWin32Header.h"

#include <itkCudaImage.h>
#include <itkCudaInPlaceImageFilter.h>

namespace rtk
{

/** @class CudaDisplacedDetectorImageFilter
 * @brief Cuda version of rtk::DisplacedDetectorImageFilter.
 *
 * Cuda version of rtk::DisplacedDetectorImageFilter.
 *
 * @see rtk::DisplacedDetectorImageFilter
 *
 * @author peter
 * @version 0.1
 */
class CudaDisplacedDetectorImageFilter :
    public  itk::CudaInPlaceImageFilter<itk::CudaImage<float,3>, itk::CudaImage<float,3>,
            rtk::DisplacedDetectorImageFilter<itk::CudaImage<float, 3> > >
{
public:
  /** Convenience typedefs **/
  typedef itk::CudaImage<float,3>                       ImageType;
  typedef rtk::DisplacedDetectorImageFilter<ImageType>  CPUWeightFilterType;

  /** Standard class typedefs. */
  typedef CudaDisplacedDetectorImageFilter                      Self;
  typedef itk::CudaInPlaceImageFilter<ImageType, ImageType,
    CPUWeightFilterType>                                        Superclass;
  typedef itk::SmartPointer<Self>                               Pointer;
  typedef itk::SmartPointer<const Self>                         ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(Self, Superclass);

protected:
  /** Standard constructor **/
  rtkcuda_EXPORT CudaDisplacedDetectorImageFilter();
  /** Destructor **/
  virtual ~CudaDisplacedDetectorImageFilter();

  virtual void GPUGenerateData();

private:
  /** purposely not implemented **/
  CudaDisplacedDetectorImageFilter(const Self&);
  /** purposely not implemented **/
  void operator=(const Self&);
};

}

#endif // RTKCUDADISPLACEDDETECTORIMAGEFILTER_H
