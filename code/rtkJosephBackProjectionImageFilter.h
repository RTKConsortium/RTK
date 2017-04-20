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

#ifndef rtkJosephBackProjectionImageFilter_h
#define rtkJosephBackProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{
namespace Functor
{
/** \class SplatWeightMultiplication
 * \brief Function to multiply the interpolation weights with the projection
 * values.
 *
 * \author Cyril Mory
 *
 * \ingroup Functions
 */
template< class TInput, class TCoordRepType, class TOutput=TCoordRepType >
class SplatWeightMultiplication
{
public:
  SplatWeightMultiplication() {};
  ~SplatWeightMultiplication() {};
  bool operator!=( const SplatWeightMultiplication & ) const
    {
    return false;
    }
  bool operator==(const SplatWeightMultiplication & other) const
    {
    return !( *this != other );
    }

  inline TOutput operator()( const TInput rayValue,
                             const double stepLengthInVoxel,
                             const double voxelSize,
                             const TCoordRepType weight) const
    {
    return rayValue * weight * voxelSize * stepLengthInVoxel;
    }
};

} // end namespace Functor


/** \class JosephBackProjectionImageFilter
 * \brief Joseph back projection.
 *
 * Performs a back projection, i.e. smearing of ray value along its path,
 * using [Joseph, IEEE TMI, 1982]. The back projector is the adjoint operator of the 
 * forward projector
 *
 * \test rtkbackprojectiontest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup Projector
 */

template <class TInputImage,
          class TOutputImage,
          class TSplatWeightMultiplication = Functor::SplatWeightMultiplication<typename TInputImage::PixelType, double, typename TOutputImage::PixelType> >
class ITK_EXPORT JosephBackProjectionImageFilter :
  public BackProjectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef JosephBackProjectionImageFilter                        Self;
  typedef BackProjectionImageFilter<TInputImage,TOutputImage>    Superclass;
  typedef itk::SmartPointer<Self>                                Pointer;
  typedef itk::SmartPointer<const Self>                          ConstPointer;
  typedef typename TInputImage::PixelType                        InputPixelType;
  typedef typename TOutputImage::PixelType                       OutputPixelType;
  typedef typename TOutputImage::RegionType                      OutputImageRegionType;
  typedef double                                                 CoordRepType;
  typedef itk::Vector<CoordRepType, TInputImage::ImageDimension> VectorType;
  typedef rtk::ThreeDCircularProjectionGeometry                  GeometryType;
  typedef typename GeometryType::Pointer                         GeometryPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(JosephBackProjectionImageFilter, BackProjectionImageFilter);

  /** Get/Set the functor that is used to multiply each interpolation value with a volume value */
  TSplatWeightMultiplication &       GetSplatWeightMultiplication() { return m_SplatWeightMultiplication; }
  const TSplatWeightMultiplication & GetSplatWeightMultiplication() const { return m_SplatWeightMultiplication; }
  void SetSplatWeightMultiplication(const TSplatWeightMultiplication & _arg)
    {
    if ( m_SplatWeightMultiplication != _arg )
      {
      m_SplatWeightMultiplication = _arg;
      this->Modified();
      }
    }


protected:
  JosephBackProjectionImageFilter() {}
  ~JosephBackProjectionImageFilter() {}

  void GenerateData() ITK_OVERRIDE;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  inline void BilinearSplat(const InputPixelType rayValue,
                            const double stepLengthInVoxel,
                            const double voxelSize,
                            OutputPixelType *pxiyi,
                            OutputPixelType *pxsyi,
                            OutputPixelType *pxiys,
                            OutputPixelType *pxsys,
                            const double x,
                            const double y,
                            const int ox,
                            const int oy);

  inline void BilinearSplatOnBorders(const InputPixelType rayValue,
                                     const double stepLengthInVoxel,
                                     const double voxelSize,
                                     OutputPixelType *pxiyi,
                                     OutputPixelType *pxsyi,
                                     OutputPixelType *pxiys,
                                     OutputPixelType *pxsys,
                                     const double x,
                                     const double y,
                                     const int ox,
                                     const int oy,
                                     const CoordRepType minx,
                                     const CoordRepType miny,
                                     const CoordRepType maxx,
                                     const CoordRepType maxy);


private:
  JosephBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                  //purposely not implemented

  /** Functor */
  TSplatWeightMultiplication m_SplatWeightMultiplication;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkJosephBackProjectionImageFilter.hxx"
#endif

#endif
