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

#ifndef rtkBackwardDifferenceDivergenceImageFilter_h
#define rtkBackwardDifferenceDivergenceImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkCastImageFilter.h>

namespace rtk
{
/** \class BackwardDifferenceDivergenceImageFilter
 * \brief Computes the backward differences divergence
 * (adjoint of the forward differences gradient) of the input image
 *
 * The exact definition of the desired divergence filter can
 * be found in Chambolle, Antonin. “An Algorithm for Total
 * Variation Minimization and Applications.” J. Math. Imaging Vis. 20,
 * no. 1–2 (January 2004): 89–97.
 *
 * \ingroup IntensityImageFilters
 */

template <typename TInputImage, typename TOutputImage = itk::Image< float, TInputImage::ImageDimension > >
class BackwardDifferenceDivergenceImageFilter :
        public itk::ImageToImageFilter< TInputImage, TOutputImage >
{
public:
    /** Extract dimension from input and output image. */
    itkStaticConstMacro(InputImageDimension, unsigned int,
                        TInputImage::ImageDimension);

    /** Convenient typedefs for simplifying declarations. */
    typedef TInputImage InputImageType;

    /** Standard class typedefs. */
    typedef BackwardDifferenceDivergenceImageFilter                Self;
    typedef itk::ImageToImageFilter< InputImageType, TOutputImage> Superclass;
    typedef itk::SmartPointer<Self>                                Pointer;
    typedef itk::SmartPointer<const Self>                          ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(BackwardDifferenceDivergenceImageFilter, ImageToImageFilter)

    /** Use the image spacing information in calculations. Use this option if you
     *  want derivatives in physical space. Default is UseImageSpacingOn. */
    void SetUseImageSpacingOn()
    { this->SetUseImageSpacing(true); }

    /** Ignore the image spacing. Use this option if you want derivatives in
        isotropic pixel space.  Default is UseImageSpacingOn. */
    void SetUseImageSpacingOff()
    { this->SetUseImageSpacing(false); }

    /** Set/Get whether or not the filter will use the spacing of the input
        image in its calculations */
    itkSetMacro(UseImageSpacing, bool);
    itkGetConstMacro(UseImageSpacing, bool);

    /** Set along which dimensions the gradient computation should be
        performed. The vector components at unprocessed dimensions are ignored */
    void SetDimensionsProcessed(bool* DimensionsProcessed);

    /** Allows to change the default boundary condition */
    void OverrideBoundaryCondition(itk::ImageBoundaryCondition< TInputImage >* boundaryCondition);

    /** Image typedef support. */
    typedef typename InputImageType::PixelType  InputPixelType;
    typedef typename InputImageType::RegionType InputImageRegionType;
    typedef typename InputImageType::SizeType   InputSizeType;
    typedef itk::CovariantVector< InputPixelType, InputImageDimension > CovariantVectorType;

protected:
    BackwardDifferenceDivergenceImageFilter();
    ~BackwardDifferenceDivergenceImageFilter();

    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    void BeforeThreadedGenerateData() ITK_OVERRIDE;

    void ThreadedGenerateData(const typename InputImageType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;

    void AfterThreadedGenerateData() ITK_OVERRIDE;

private:
    BackwardDifferenceDivergenceImageFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    bool                              m_UseImageSpacing;
    typename TInputImage::SpacingType m_InvSpacingCoeffs;

    // list of the dimensions along which the divergence has
    // to be computed. The components on other dimensions
    // are ignored for performance, but the gradient filter
    // sets them to zero anyway
    bool m_DimensionsProcessed[TInputImage::ImageDimension];

    // The default is ConstantBoundaryCondition, but this behavior sometimes needs to be overriden
    itk::ImageBoundaryCondition< TInputImage, TInputImage >* m_BoundaryCondition;
    // If so, do not perform boundary processing in AfterThreadedGenerateData
    bool                                                     m_IsBoundaryConditionOverriden;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkBackwardDifferenceDivergenceImageFilter.hxx"
#endif

#endif //__rtkBackwardDifferenceDivergenceImageFilter__
