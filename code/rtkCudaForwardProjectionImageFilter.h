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

#ifndef __rtkCudaForwardProjectionImageFilter_h
#define __rtkCudaForwardProjectionImageFilter_h

#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkCudaForwardProjectionImageFilter.hcu"
#include "itkCudaInPlaceImageFilter.h"
#include "itkCudaUtil.h"
#include "itkCudaKernelManager.h"
#include "rtkWin32Header.h"

#include "itkImage.h"

/** \class CudaForwardProjectionImageFilter
 * \brief Trilinear interpolation forward projection implemented in CUDA
 *
 * CudaForwardProjectionImageFilter is similar to
 * JosephForwardProjectionImageFilter, except it uses a
 * fixed step between sampling points instead of placing these
 * sampling points only on the main direction slices.
 *
 * The code was developed based on the file tt_project_ray_gpu_kernels.cu of
 * NiftyRec (http://sourceforge.net/projects/niftyrec/) which is distributed under a BSD
 * license. See COPYRIGHT.TXT.
 *
 * \author Marc Vila, updated by Simon Rit and Cyril Mory
 *
 * \ingroup Projector CudaImageToImageFilter
 */

namespace rtk
{

namespace Functor
{

class CudaInterpolationWeightMultiplication
{
public:
  CudaInterpolationWeightMultiplication() {};
  ~CudaInterpolationWeightMultiplication() {};
  bool operator!=( const CudaInterpolationWeightMultiplication & ) const {
    return false;
  }
  bool operator==(const CudaInterpolationWeightMultiplication & other) const {
    return !( *this != other );
  }

  inline double operator()( const rtk::ThreadIdType threadId,
                            const double stepLengthInVoxel,
                            const double weight,
                            const float *p,
                            const int i)
  {
    return 0.;
  }

  std::vector<double>* GetInterpolationWeights() { return 0; }
};

template< class TInput, class TOutput >
class CudaProjectedValueAccumulation
{
public:
  typedef itk::Vector<double, 3> VectorType;
  typedef itk::Image<double, 2>  MaterialMuImageType;

  CudaProjectedValueAccumulation() {};
  ~CudaProjectedValueAccumulation() {};

  virtual CudaAccumulationParameters* GetCudaParameters() = 0;
};

template< class TInput, class TOutput >
class CudaProjectedValueAccumulationOriginal : public CudaProjectedValueAccumulation<TInput, TOutput>
{
public:
  typedef CudaProjectedValueAccumulation<TInput, TOutput> Superclass;
  typedef typename Superclass::MaterialMuImageType        MaterialMuImageType;
  
  CudaProjectedValueAccumulationOriginal()
  {
    // fake mu initialization
    typename MaterialMuImageType::SizeType size;
    size[0] = 20;
    size[1] = 10;
    typename MaterialMuImageType::RegionType region;
    region.SetSize(size);
    m_MaterialMu = MaterialMuImageType::New();
    m_MaterialMu->SetRegions(region);
    m_MaterialMu->Allocate();
  };
  ~CudaProjectedValueAccumulationOriginal() {};
  
  CudaAccumulationParameters* GetCudaParameters()
  {
    CudaAccumulationParameters* params = new CudaAccumulationParameters();
    params->projectionType = ORIGINAL;
    params->matSize = m_MaterialMu->GetLargestPossibleRegion().GetSize()[0];
    return params;
  }
  
  // fake mu
  typename MaterialMuImageType::Pointer  m_MaterialMu;
};

template< class TInput, class TOutput >
class CudaProjectedValueAccumulationPrimary : public CudaProjectedValueAccumulation<TInput, TOutput>
{
public:
  typedef CudaProjectedValueAccumulation<TInput, TOutput> Superclass;
  typedef typename Superclass::MaterialMuImageType        MaterialMuImageType;

  CudaProjectedValueAccumulationPrimary()
  {
    // fake mu initialization
    typename MaterialMuImageType::SizeType size;
    size[0] = 20;
    size[1] = 10;
    typename MaterialMuImageType::RegionType region;
    region.SetSize(size);
    m_MaterialMu = MaterialMuImageType::New();
    m_MaterialMu->SetRegions(region);
    m_MaterialMu->Allocate();
    itk::ImageRegionIterator< MaterialMuImageType > it(m_MaterialMu, region);
    for(unsigned int e = 0; e < size[1]; e++)
      {
      for(unsigned int i = 0; i < size[0]-1; i++)
        {
        it.Set(0.1);
        ++it;
        }
      it.Set(0.);
      ++it;
      }
    
    // fake energy initialization
    m_EnergyWeightList = new std::vector<double>();
    for (int i = 0; i < (int)size[1]; i++)
      {
      m_EnergyWeightList->push_back(i);
      }
  };
  ~CudaProjectedValueAccumulationPrimary()
  {
    delete m_EnergyWeightList;
  };
  
  CudaAccumulationParameters* GetCudaParameters()
  {
    CudaAccumulationParameters* params = new CudaAccumulationParameters();

    params->projectionType = PRIMARY;
    params->matSize = m_MaterialMu->GetLargestPossibleRegion().GetSize()[0];
    params->energySize = m_MaterialMu->GetLargestPossibleRegion().GetSize()[1];
    params->mu = new float[params->matSize*params->energySize];
    
    double* p = m_MaterialMu->GetBufferPointer();
    float* pCopy = params->mu;
    for (int i = 0; i < params->matSize*params->energySize; i++)
      {
      *pCopy = *p;
      ++p; ++pCopy;
      }
    
    params->energyWeights = new float[m_EnergyWeightList->size()];
    for(unsigned int i = 0; i < m_EnergyWeightList->size(); i++)
      {
      params->energyWeights[i] = (*m_EnergyWeightList)[i];
      }

    return params;
  }

  typename MaterialMuImageType::Pointer m_MaterialMu;
  std::vector<double>         *m_EnergyWeightList;
};

template< class TInput, class TOutput >
class CudaProjectedValueAccumulationCompton : public CudaProjectedValueAccumulation<TInput, TOutput>
{
public:
  typedef CudaProjectedValueAccumulation<TInput, TOutput> Superclass;
  typedef typename Superclass::MaterialMuImageType        MaterialMuImageType;
  typedef typename Superclass::VectorType                 VectorType;
  
  CudaProjectedValueAccumulationCompton()
  {
    // fake mu initialization
    typename MaterialMuImageType::SizeType size;
    size[0] = 20;
    size[1] = 10;
    typename MaterialMuImageType::RegionType region;
    region.SetSize(size);
    m_MaterialMu = MaterialMuImageType::New();
    m_MaterialMu->SetRegions(region);
    m_MaterialMu->Allocate();
    itk::ImageRegionIterator< MaterialMuImageType > it(m_MaterialMu, region);
    for(unsigned int e = 0; e < size[1]; e++)
      {
      for(unsigned int i = 0; i < size[0]-1; i++)
        {
        it.Set(0.1);
        ++it;
        }
      it.Set(0.);
      ++it;
      }

    // fake variables initialization
    for(int i = 0; i < 3; i++)
      {
      m_Direction[i] = (float)i;
      }
    for(int i = 0; i < 3; i++)
      {
      m_DetectorOrientationTimesPixelSurface[i] = (float)i;
      }
    m_InvWlPhoton = 1.;
    m_E0m = 1.;
    m_eRadiusOverCrossSectionTerm = 1.;
    m_Energy = 1.;
  };
  ~CudaProjectedValueAccumulationCompton() {};
  
  CudaAccumulationParameters* GetCudaParameters()
  {
    CudaAccumulationParameters* params = new CudaAccumulationParameters();

    params->projectionType = COMPTON;
    params->matSize = m_MaterialMu->GetLargestPossibleRegion().GetSize()[0];
    params->energySize = m_MaterialMu->GetLargestPossibleRegion().GetSize()[1];
    params->mu = new float[params->matSize*params->energySize];

    double* p = m_MaterialMu->GetBufferPointer();
    float* pCopy = params->mu;
    for (int i = 0; i < params->matSize*params->energySize; i++)
      {
      *pCopy = *p;
      ++p; ++pCopy;
      }

    params->energy_spacing = m_MaterialMu->GetSpacing()[1];
    params->invWlPhoton = m_InvWlPhoton;
    params->e0m = m_E0m;
    params->eRadiusOverCrossSectionTerm = m_eRadiusOverCrossSectionTerm;
    params->energy = m_Energy;
    for(int i = 0; i < 3; i++)
      {
      params->direction[i] = m_Direction[i];
      }
    for(int i = 0; i < 3; i++)
      {
      params->detectorOrientationTimesPixelSurface[i] = m_DetectorOrientationTimesPixelSurface[i];
      }

    return params;
  }
  
  typename MaterialMuImageType::Pointer m_MaterialMu;
  VectorType m_Direction;
  double m_Energy;
  double m_E0m;
  double m_InvWlPhoton;
  double m_eRadiusOverCrossSectionTerm;
  VectorType m_DetectorOrientationTimesPixelSurface;

  /*
  unsigned int               m_Z;
  GateEnergyResponseFunctor *m_ResponseDetector;

  // Compton data
  G4VEMDataSet* m_ScatterFunctionData;
  G4VCrossSectionHandler* m_CrossSectionHandler;
  */
};

} // end namespace Functor

/** Create a helper Cuda Kernel class for CudaImageOps */
itkCudaKernelClassMacro(rtkCudaForwardProjectionImageFilterKernel);

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication = Functor::CudaInterpolationWeightMultiplication,
          class TProjectedValueAccumulation        = Functor::CudaProjectedValueAccumulationCompton<typename TInputImage::PixelType, typename TOutputImage::PixelType>
          >
class ITK_EXPORT CudaForwardProjectionImageFilter :
  public itk::CudaInPlaceImageFilter< TInputImage, TOutputImage,
  ForwardProjectionImageFilter< TInputImage, TOutputImage > >
{
public:
  /** Standard class typedefs. */
  typedef CudaForwardProjectionImageFilter                                    Self;
  typedef ForwardProjectionImageFilter<TInputImage, TOutputImage>             Superclass;
  typedef itk::CudaInPlaceImageFilter<TInputImage, TOutputImage, Superclass > GPUSuperclass;
  typedef itk::SmartPointer<Self>                                             Pointer;
  typedef itk::SmartPointer<const Self>                                       ConstPointer;
  typedef typename TOutputImage::RegionType                                   OutputImageRegionType;
  typedef itk::Vector<float,3>                                                VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaForwardProjectionImageFilter, ImageToImageFilter);

  /** Get/Set the functor that is used to multiply each interpolation value with a volume value */
  TInterpolationWeightMultiplication &       GetInterpolationWeightMultiplication() { return m_InterpolationWeightMultiplication; }
  const TInterpolationWeightMultiplication & GetInterpolationWeightMultiplication() const { return m_InterpolationWeightMultiplication; }
  void SetInterpolationWeightMultiplication(const TInterpolationWeightMultiplication & _arg)
    {
    if ( m_InterpolationWeightMultiplication != _arg )
      {
      m_InterpolationWeightMultiplication = _arg;
      this->Modified();
      }
    }

  /** Get/Set the functor that is used to accumulate values in the projection image after the ray
   * casting has been performed. */
  TProjectedValueAccumulation &       GetProjectedValueAccumulation() { return m_ProjectedValueAccumulation; }
  const TProjectedValueAccumulation & GetProjectedValueAccumulation() const { return m_ProjectedValueAccumulation; }
  void SetProjectedValueAccumulation(const TProjectedValueAccumulation & _arg)
    {
    if ( m_ProjectedValueAccumulation != _arg )
      {
      m_ProjectedValueAccumulation = _arg;
      this->Modified();
      }
    }

protected:
  rtkcuda_EXPORT CudaForwardProjectionImageFilter() {};
  ~CudaForwardProjectionImageFilter() {};
  
  void GPUGenerateData();

private:
  //purposely not implemented
  CudaForwardProjectionImageFilter(const Self&);
  void operator=(const Self&);

  TInterpolationWeightMultiplication m_InterpolationWeightMultiplication;
  TProjectedValueAccumulation        m_ProjectedValueAccumulation;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkCudaForwardProjectionImageFilter.txx"
#endif

#endif
