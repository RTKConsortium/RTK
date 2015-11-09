/*=========================================================================
*
*  Copyright Insight Software Consortium
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
#ifndef __itkCudaImage_h
#define __itkCudaImage_h

#include "itkImage.h"
#include "itkCudaImageDataManager.h"
#include "itkVersion.h"
#include "itkObjectFactoryBase.h"

namespace itk
{
/** \class CudaImage
 *  \brief Templated n-dimensional image class for the Cuda.
 *
 * Derived from itk Image class to use with Cuda image filters.
 * This class manages both CPU and Cuda memory implicitly, and
 * can be used with non-Cuda itk filters as well. Memory transfer
 * between CPU and Cuda is done automatically and implicitly.
 *
 * \ingroup ITKCudaCommon
 */
template <class TPixel, unsigned int VImageDimension = 2>
class ITK_EXPORT CudaImage : public Image<TPixel,VImageDimension>
{
public:
  typedef CudaImage                     Self;
  typedef Image<TPixel,VImageDimension> Superclass;
  typedef SmartPointer<Self>            Pointer;
  typedef SmartPointer<const Self>      ConstPointer;
  typedef WeakPointer<const Self>       ConstWeakPointer;

  itkNewMacro(Self);

  itkTypeMacro(CudaImage, Image);

  itkStaticConstMacro(ImageDimension, unsigned int, VImageDimension);

  typedef typename Superclass::PixelType         PixelType;
  typedef typename Superclass::ValueType         ValueType;
  typedef typename Superclass::InternalPixelType InternalPixelType;
  typedef typename Superclass::IOPixelType       IOPixelType;
  typedef typename Superclass::DirectionType     DirectionType;
  typedef typename Superclass::SpacingType       SpacingType;
  typedef typename Superclass::PixelContainer    PixelContainer;
  typedef typename Superclass::SizeType          SizeType;
  typedef typename Superclass::IndexType         IndexType;
  typedef typename Superclass::OffsetType        OffsetType;
  typedef typename Superclass::RegionType        RegionType;
  typedef typename PixelContainer::Pointer       PixelContainerPointer;
  typedef typename PixelContainer::ConstPointer  PixelContainerConstPointer;
  typedef typename Superclass::AccessorType      AccessorType;
   
  typedef unsigned long                         ModifiedTimeType;
  typedef DefaultPixelAccessorFunctor< Self >   AccessorFunctorType;

  typedef NeighborhoodAccessorFunctor< Self >   NeighborhoodAccessorFunctorType;

  /**
   * example usage:
   * typedef typename ImageType::template Rebind< float >::Type OutputImageType;
   *
   */
  template <typename UPixelType, unsigned int UImageDimension = VImageDimension>
  struct Rebind
    {
      typedef itk::CudaImage<UPixelType, UImageDimension>  Type;
    };

  //
  // Allocate CPU and Cuda memory space
  //
#if ITK_VERSION_MAJOR < 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR < 6)
  void Allocate();
#else
  void Allocate(bool initializePixels = false);
#endif

  virtual void Initialize();

  void FillBuffer(const TPixel & value);

  void SetPixel(const IndexType & index, const TPixel & value);

  const TPixel & GetPixel(const IndexType & index) const;

  TPixel & GetPixel(const IndexType & index);

  const TPixel & operator[](const IndexType & index) const;

  TPixel & operator[](const IndexType & index);

  /** Explicit synchronize CPU/Cuda buffers */
  void UpdateBuffers();

  //
  // Get CPU buffer pointer
  //
  TPixel* GetBufferPointer();

  const TPixel * GetBufferPointer() const;

  /** Return the Pixel Accessor object */
  AccessorType GetPixelAccessor(void)
  {
    m_DataManager->SetGPUBufferDirty();
    return Superclass::GetPixelAccessor();
  }

  /** Return the Pixel Accesor object */
  const AccessorType GetPixelAccessor(void) const
  {
    m_DataManager->UpdateCPUBuffer();
    return Superclass::GetPixelAccessor();
  }

  /** Return the NeighborhoodAccessor functor */
  NeighborhoodAccessorFunctorType GetNeighborhoodAccessor()
  {
    m_DataManager->SetGPUBufferDirty();
    //return Superclass::GetNeighborhoodAccessor();
    return NeighborhoodAccessorFunctorType();
  }

  /** Return the NeighborhoodAccessor functor */
  const NeighborhoodAccessorFunctorType GetNeighborhoodAccessor() const
  {
    m_DataManager->UpdateCPUBuffer();
    //return Superclass::GetNeighborhoodAccessor();
    return NeighborhoodAccessorFunctorType();
  }

  void SetPixelContainer(PixelContainer *container);

  /** Return a pointer to the container. */
  PixelContainer * GetPixelContainer()
  {
    m_DataManager->SetGPUBufferDirty(); 
    return Superclass::GetPixelContainer();
  }

  const PixelContainer * GetPixelContainer() const
  {
    m_DataManager->UpdateCPUBuffer();
    return Superclass::GetPixelContainer();
  }

  itkGetObjectMacro(DataManager, CudaImageDataManager< CudaImage >);

  CudaDataManager::Pointer GetCudaDataManager() const;
  
  /** Overload the SetBufferedRegion function because if the size changes we need
   *  to invalidated the GPU buffer */
  void SetBufferedRegion(const RegionType & region);

  /* Override DataHasBeenGenerated() in DataObject class.
   * We need this because CPU time stamp is always bigger
   * than Cuda's. That is because Modified() is called at
   * the end of each filter in the pipeline so although we
   * increment Cuda's time stamp in CudaGenerateData() the
   * CPU's time stamp will be increased after that.
   */
  void DataHasBeenGenerated()
  {
    Superclass::DataHasBeenGenerated();
    if (m_DataManager->IsCPUBufferDirty())
      {
      m_DataManager->Modified();
      }
  }

  /** Graft the data and information from one CudaImage to another. */
  virtual void Graft(const DataObject *data);
  
protected:
  CudaImage();
  virtual ~CudaImage();

private:

  // functions that are purposely not implemented
  CudaImage(const Self&);
  void operator=(const Self&);
  
  typename CudaImageDataManager< CudaImage >::Pointer m_DataManager;
};

class CudaImageFactory : public itk::ObjectFactoryBase
{
public:
  typedef CudaImageFactory              Self;
  typedef itk::ObjectFactoryBase        Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion() const {
    return ITK_SOURCE_VERSION;
  }
  const char* GetDescription() const {
    return "A Factory for CudaImage";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaImageFactory, itk::ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
  {
    CudaImageFactory::Pointer factory = CudaImageFactory::New();

    itk::ObjectFactoryBase::RegisterFactory(factory);
  }

private:
  CudaImageFactory(const Self&); //purposely not implemented
  void operator=(const Self&);  //purposely not implemented

#define OverrideImageTypeMacro(pt,dm)    this->RegisterOverride(\
    typeid(itk::Image<pt,dm>).name(), \
    typeid(itk::CudaImage<pt,dm>).name(), \
    "Cuda Image Override", \
    true, \
    itk::CreateObjectFunction<CudaImage<pt,dm> >::New())

  CudaImageFactory()
  {
    if (IsCudaAvailable())
      {
      // 1/2/3D
      OverrideImageTypeMacro(unsigned char, 1);
      OverrideImageTypeMacro(signed char,  1);
      OverrideImageTypeMacro(int, 1);
      OverrideImageTypeMacro(unsigned int, 1);
      OverrideImageTypeMacro(float, 1);
      OverrideImageTypeMacro(double, 1);

      OverrideImageTypeMacro(unsigned char, 2);
      OverrideImageTypeMacro(signed char, 2);
      OverrideImageTypeMacro(int, 2);
      OverrideImageTypeMacro(unsigned int, 2);
      OverrideImageTypeMacro(float, 2);
      OverrideImageTypeMacro(double, 2);

      OverrideImageTypeMacro(unsigned char, 3);
      OverrideImageTypeMacro(signed char, 3);
      OverrideImageTypeMacro(int, 3);
      OverrideImageTypeMacro(unsigned int, 3);
      OverrideImageTypeMacro(float, 3);
      OverrideImageTypeMacro(double, 3);
      }
  }

};

template <class T>
class CudaTraits
{
public:
  typedef T Type;
};

template <class TPixelType, unsigned int NDimension>
class CudaTraits< Image< TPixelType, NDimension > >
{
public:
  typedef CudaImage<TPixelType,NDimension> Type;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCudaImage.hxx"
#endif

#endif
