//
#ifndef RTKCUDAPARKERSHORTSCANIMAGEFILTER_H
#define RTKCUDAPARKERSHORTSCANIMAGEFILTER_H

#include "rtkParkerShortScanImageFilter.h"
#include "rtkWin32Header.h"

#include <itkCudaImage.h>
#include <itkCudaInPlaceImageFilter.h>

namespace rtk
{

/** @class CudaParkerShortScanImageFilter
 * @brief Cuda version of rtk::ParkerShortScanImageFilter.
 *
 * Cuda version of rtk::ParkerShortScanImageFilter.
 *
 * @see rtk::ParkerShortScanImageFilter
 *
 * @author peter
 * @version 0.1
 */
class CudaParkerShortScanImageFilter :
    public  itk::CudaInPlaceImageFilter<itk::CudaImage<float,3>, itk::CudaImage<float,3>,
            rtk::ParkerShortScanImageFilter<itk::CudaImage<float, 3> > >
{
public:
  /** Convenience typedefs **/
  typedef itk::CudaImage<float,3>                     ImageType;
  typedef rtk::ParkerShortScanImageFilter<ImageType>  CPUWeightFilterType;

  /** Standard class typedefs. */
  typedef CudaParkerShortScanImageFilter                        Self;
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
  rtkcuda_EXPORT CudaParkerShortScanImageFilter();
  /** Destructor **/
  virtual ~CudaParkerShortScanImageFilter();

  virtual void GPUGenerateData();

private:
  /** purposely not implemented **/
  CudaParkerShortScanImageFilter(const Self&);
  /** purposely not implemented **/
  void operator=(const Self&);
};

}

#endif // RTKCUDAPARKERSHORTSCANIMAGEFILTER_H
