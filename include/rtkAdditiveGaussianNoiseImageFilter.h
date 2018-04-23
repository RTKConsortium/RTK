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

/*Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: $
  Language:  C++
  Date:      $Date: 2007-08-31 22:17:25 +0200 (Fri, 31 Aug 2007) $
  Version:   $Revision: 2 $
  Author:    Gavin Baker <gavinb@cs.mu.oz.au>

  Copyright (c) 2004 Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef rtkAdditiveGaussianNoiseImageFilter_h
#define rtkAdditiveGaussianNoiseImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkNormalVariateGenerator.h>
#include <itkUnaryFunctorImageFilter.h>

#include "rtkMacro.h"

namespace rtk
{

/** \class NormalVariateNoiseFunctor
*
* \brief Pixel functor that adds Gaussian noise.
*
* \author Gavin Baker: gavinb at cs_mu_oz_au
*/
template < class TPixel >
class NormalVariateNoiseFunctor
{
public:

  NormalVariateNoiseFunctor()
  {
    m_Mean = 0.0;
    m_StandardDeviation = 1.0;

    m_Generator = itk::Statistics::NormalVariateGenerator::New();

    this->SetSeed( 42 );
  }

  float GetMean() const
  {
    return m_Mean;
  }

  void SetMean( float mean )
  {
    m_Mean = mean;
  }

  float GetStandardDeviation() const
  {
    return m_StandardDeviation;
  }

  void SetStandardDeviation( float stddev )
  {
    m_StandardDeviation = stddev;
  }

  void SetSeed( unsigned long seed )
  {
    m_Generator->Initialize( seed );
  }
  
  void SetOutputMinimum( TPixel min ) {
    m_OutputMinimum = min;
  }

  void SetOutputMaximum( TPixel max ) {
    m_OutputMaximum = max;
  }

  TPixel GetOutputMinimum() const {
    return m_OutputMinimum;
  }

  TPixel GetOutputMaximum() const {
    return m_OutputMaximum;
  }

  TPixel operator()( TPixel input )
  {
    // Get the minimum and maximum output values
    static const float min = static_cast<float>( m_OutputMinimum );
    static const float max = static_cast<float>( m_OutputMaximum );

    // Compute the output
    float output = static_cast<float>( input ) +
      m_Mean + m_StandardDeviation * m_Generator->GetVariate();

    // Clamp the output value in valid range
    output = ( output < min ? min : output );
    output = ( output > max ? max : output );

    return static_cast< TPixel > ( output );
  }

private:
  TPixel                                           m_OutputMinimum;
  TPixel                                           m_OutputMaximum;
  float                                            m_Mean;
  float                                            m_StandardDeviation;
  itk::Statistics::NormalVariateGenerator::Pointer m_Generator;
};


/** \class AdditiveGaussianNoiseImageFilter
 * \brief Adds Gaussian noise to the input image
 *
 * Adds noise to the input image according to a Gaussian normal variate
 * distribution.  The user supplies the mean \f$\bar{x}\f$ and standard
 * deviation \f$\sigma\f$, such that the output is given by:
 *
 * \f[
 *     v_{out} = v_{in} + \bar{x} + \sigma * G(d)
 * \f]
 *
 * where G() is the Gaussian generator and d is the seed.  A particular seed
 * can be specified in order to perform repeatable tests.
 *
 * \test rtkrampfiltertest.cxx
 *
 * \author Gavin Baker: gavinb at cs_mu_oz_au
 *
 * \ingroup ImageToImageFilter
 */
template <class TInputImage >
class ITK_EXPORT AdditiveGaussianNoiseImageFilter :
    public itk::ImageToImageFilter< TInputImage, TInputImage >
{
public:
  /** Standard class typedefs. */
  typedef AdditiveGaussianNoiseImageFilter                  Self;
  typedef itk::ImageToImageFilter<TInputImage, TInputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory */
  itkNewMacro(Self);  

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdditiveGaussianNoiseImageFilter, ImageToImageFilter);

  /** Superclass typedefs. */
  typedef typename Superclass::OutputImageRegionType OutputImageRegionType;
  typedef typename Superclass::OutputImagePointer    OutputImagePointer;

  /** Some convenient typedefs. */
  typedef TInputImage                           InputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::PixelType    InputImagePixelType;
  typedef typename InputImageType::PixelType    InputPixelType;

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension);

  // virtual void GenerateOutputInformation();

  void GenerateData() ITK_OVERRIDE;

  // Accessor & Mutator methods

  /**
   *    Specifies the average noise added to the image per pixel.
   *    The default is 0.
   */
  void SetMean( float mean )
  {
    m_NoiseFilter->GetFunctor().SetMean( mean );
    this->Modified();
  }

  /**
   *    Returns the average noise added to the image per pixel.
   *    The default is 0.
   */
  float GetMean() const
  {
    return m_NoiseFilter->GetFunctor().GetMean();
  }

  /**
   *    Specifies the standard deviation of the noise added to the image.
   *    The default is 1.
   */
  void SetStandardDeviation( float stddev )
  {
    m_NoiseFilter->GetFunctor().SetStandardDeviation( stddev );
    this->Modified();
  }

  /**
   *    Returns the standard deviation of the noise added to the image.
   *    The default is 1.
   */
  float GetStandardDeviation() const
  {
    return m_NoiseFilter->GetFunctor().GetStandardDeviation();
  }

  /**
   *    Specifies the seed for the normal variate generator.  The same seed
   *    will produce the same pseduo-random sequence, which can be used to
   *    reproduce results.  For a higher dose of entropy, initialise with
   *    the current system time (in ms).
   */
  void SetSeed( unsigned long seed )
  {
    m_NoiseFilter->GetFunctor().SetSeed( seed );
    this->Modified();
  }

  /** Set the minimum output value. */
  void SetOutputMinimum( InputImagePixelType min )
  {
    if( min == m_NoiseFilter->GetFunctor().GetOutputMinimum() )
      {
      return;
      }
    m_NoiseFilter->GetFunctor().SetOutputMinimum( min );
    this->Modified();
  }

  /** Get the minimum output value. */
  InputImagePixelType GetOutputMinimum( )
  {
    return m_NoiseFilter->GetFunctor().GetOutputMinimum();
  }

  /** Set the maximum output value. */
  void SetOutputMaximum( InputImagePixelType max )
  {
    if( max == m_NoiseFilter->GetFunctor().GetOutputMaximum() ) 
      {
      return;
      }
    m_NoiseFilter->GetFunctor().SetOutputMaximum( max );
    this->Modified();
  }

  /** Get the maximum output value. */
  InputImagePixelType GetOutputMaximum( )
  {
    return m_NoiseFilter->GetFunctor().GetOutputMaximum();
  }

protected:

  AdditiveGaussianNoiseImageFilter();

  void PrintSelf(std::ostream& os, itk::Indent indent) const ITK_OVERRIDE;

private:

  AdditiveGaussianNoiseImageFilter(const Self&);  // intentionally not implemented
  void operator=(const Self&);      // intentionally not implemented

public:

  typedef itk::UnaryFunctorImageFilter< InputImageType, InputImageType,
                                       NormalVariateNoiseFunctor< typename InputImageType::PixelType > >
                                       NoiseFilterType;

private:

  typename NoiseFilterType::Pointer m_NoiseFilter;
};

} /* end namespace rtk */

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkAdditiveGaussianNoiseImageFilter.hxx"
#endif

#endif /* rtkAdditiveGaussianNoiseImageFilter_h */
