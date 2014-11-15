/*=========================================================================
 *
 *  Copyright Insight Software Consortium & RTK Consortium
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
#ifndef __srtkInterpolator_h
#define __srtkInterpolator_h

#include "srtkCommon.h"
#include <ostream>


namespace rtk
{
namespace simple
{

enum InterpolatorEnum {
  /** \brief Nearest-neighbor interpolation
   * \sa itk::NearestNeighborInterpolateImageFunction
   */
  srtkNearestNeighbor = 1,

  /** \brief N-D linear interpolation
   * \sa itk::LinearInterpolateImageFunction
   */
  srtkLinear = 2,

  /** \brief B-Spline of order 3 interpolation
   * \sa itk::BSplineInterpolateImageFunction
   */
  srtkBSpline = 3,

  /** \brief Gaussian interpolation
   *
   * Sigma is set to 0.8 input pixels and alpha is 4.0
   *
   * \sa itk::GaussianInterpolateImageFunction
   */
  srtkGaussian = 4,

  /** \brief Smoothly interpolate multi-label images
   *
   * Sigma is set to 1.0 input pixels and alpha is 1.0
   *
   * \sa itk:LabelImageGaussianInterpolateImageFunction
   */
  srtkLabelGaussian = 5,

  /** \brief Windowed sinc interpolation
   *
   * \f[ w(x) = 0.54 + 0.46 cos(\frac{\pi x}{m} ) \f]
   *
   * \sa itk::WindowedSincInterpolateImageFunction
   * \sa itk::Function::HammingWindowFunction
   */
  srtkHammingWindowedSinc = 6,

  /** \brief Windowed sinc interpolation
   *
   * \f[ w(x) = cos(\frac{\pi x}{2 m} ) \f]
   *
   * \sa itk::WindowedSincInterpolateImageFunction
   * \sa itk::Function::CosineWindowFunction
   */
  srtkCosineWindowedSinc = 7,

  /** \brief Windowed sinc interpolation
   *
   * \f[ w(x) = 1 - ( \frac{x^2}{m^2} ) \f]
   *
   * \sa itk::WindowedSincInterpolateImageFunction
   * \sa itk::Function::WelchWindowFunction
   */
  srtkWelchWindowedSinc = 8,

  /** \brief Windowed sinc interpolation
   *
   * \f[ w(x) = \textrm{sinc} ( \frac{x}{m} ) \f]
   *
   * \sa itk::WindowedSincInterpolateImageFunction
   * \sa itk::Function::LanczosWindowFunction
   */
  srtkLanczosWindowedSinc = 9,

  /** \brief Windowed sinc interpolation
   *
   * \f[ w(x) = 0.42 + 0.5 cos(\frac{\pi x}{m}) + 0.08 cos(\frac{2 \pi x}{m}) \f]
   *
   * \sa itk::WindowedSincInterpolateImageFunction
   * \sa itk::Function::BlackmanWindowFunction
   */
  srtkBlackmanWindowedSinc = 10
};

#ifndef SWIG
/**
 * Convert Interpolator enum to a string for printing etc..
 */
SRTKCommon_EXPORT std::ostream& operator<<(std::ostream& os, const InterpolatorEnum i);
#endif

} // end namespace simple
} // end namespace rtk


#endif //__srtkInterpolator_h
