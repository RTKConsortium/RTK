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
#ifndef rtkImageRegionSplitterArbitraryDimension_h
#define rtkImageRegionSplitterArbitraryDimension_h

#include <itkImageRegionSplitterBase.h>

namespace rtk
{

/** \class ImageRegionSplitterArbitraryDimension
 * \brief Divide an image region along the dimension specified by SetSplitAxis
 *
 * ImageRegionSplitterArbitraryDimension divides an ImageRegion into smaller regions.
 * ImageRegionSplitterArbitraryDimension is the default splitter for many situations.
 *
 * This ImageRegionSplitterArbitraryDimension class divides a region along the
 * outermost or slowest dimension. If the outermost dimension has size
 * 1 (i.e. a volume with a single slice), the ImageRegionSplitter will
 * divide the region along the next outermost dimension. If that
 * dimension has size 1, the process continues with the next outermost
 * dimension.
 *
 * \sa ImageRegionSplitterDirection
 *
 * \ingroup ITKSystemObjects
 * \ingroup DataProcessing
 * \ingroup ITKCommon
 */

class ITKCommon_EXPORT ImageRegionSplitterArbitraryDimension
  :public itk::ImageRegionSplitterBase
{
public:
  /** Standard class typedefs. */
  typedef ImageRegionSplitterArbitraryDimension Self;
  typedef itk::ImageRegionSplitterBase          Superclass;
  typedef itk::SmartPointer< Self >             Pointer;
  typedef itk::SmartPointer< const Self >       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageRegionSplitterArbitraryDimension, ImageRegionSplitterBase);

  /** Set/Get methods for the split axis */
  itkSetMacro(SplitAxis,unsigned int)
  itkGetMacro(SplitAxis,unsigned int)

protected:
  ImageRegionSplitterArbitraryDimension();

  virtual unsigned int GetNumberOfSplitsInternal( unsigned int dim,
                                                  const itk::IndexValueType regionIndex[],
                                                  const itk::SizeValueType regionSize[],
                                                  unsigned int requestedNumber ) const ITK_OVERRIDE;

  virtual unsigned int GetSplitInternal( unsigned int dim,
                                         unsigned int i,
                                         unsigned int numberOfPieces,
                                         itk::IndexValueType regionIndex[],
                                         itk::SizeValueType regionSize[] ) const ITK_OVERRIDE;

  unsigned int m_SplitAxis;

private:
  ITK_DISALLOW_COPY_AND_ASSIGN(ImageRegionSplitterArbitraryDimension);
};
} // end namespace rtk

#endif
