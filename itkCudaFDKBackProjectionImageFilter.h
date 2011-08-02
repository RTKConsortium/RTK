#ifndef __itkCudaFDKBackProjectionImageFilter_h
#define __itkCudaFDKBackProjectionImageFilter_h

#include "itkFDKBackProjectionImageFilter.h"

namespace itk
{

class ITK_EXPORT CudaFDKBackProjectionImageFilter :
  public FDKBackProjectionImageFilter< itk::Image<float,3>, itk::Image<float,3> >
{
public:
  /** Standard class typedefs. */
  typedef itk::Image<float,3>                                ImageType;
  typedef CudaFDKBackProjectionImageFilter                   Self;
  typedef FDKBackProjectionImageFilter<ImageType, ImageType> Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef SmartPointer<const Self>                           ConstPointer;

  typedef ImageType::RegionType            OutputImageRegionType;
  typedef itk::Image<float, 2>             ProjectionImageType;
  typedef ProjectionImageType::Pointer     ProjectionImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaFDKBackProjectionImageFilter, ImageToImageFilter);

  /** Function to allocate memory on device */
  void InitDevice();

  /** Function to synchronize memory from device to host and free device memory */
  void CleanUpDevice();

protected:
  CudaFDKBackProjectionImageFilter();
  virtual ~CudaFDKBackProjectionImageFilter() {};

  void GenerateData();

private:
  CudaFDKBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                   //purposely not implemented

  int        m_VolumeDimension[3];
  int        m_ProjectionDimension[2];
  float *    m_DeviceVolume;
  float *    m_DeviceProjection;
  float *    m_DeviceMatrix;
};

} // end namespace itk

#endif
