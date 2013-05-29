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

#ifndef __rtkProjectionsReader_h
#define __rtkProjectionsReader_h

// ITK
#include <itkImageSource.h>
#include <itkImageIOFactory.h>

// Standard lib
#include <vector>
#include <string>

namespace rtk
{

/** \class ProjectionsReader
 *
 * This is the universal projections reader of rtk (raw data converted to
 * understandable values, e.g. attenuation). Currently handles his (Elekta
 * Synergy), hnd (Varian OBI), tiff. For all other ITK file formats, it is
 * assumed that the attenuation is directly passed and there is no processing,
 * only the reading.
 *
 * \test rtkedftest.cxx, rtkelektatest.cxx, rtkimagxtest.cxx, 
 * rtkdigisenstest.cxx, rtkxradtest.cxx, rtkvariantest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ImageSource
 */
template <class TOutputImage>
class ITK_EXPORT ProjectionsReader : public itk::ImageSource<TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ProjectionsReader              Self;
  typedef itk::ImageSource<TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>        Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ProjectionsReader, itk::ImageSource);

  /** Some convenient typedefs. */
  typedef TOutputImage                         OutputImageType;
  typedef typename OutputImageType::Pointer    OutputImagePointer;
  typedef typename OutputImageType::RegionType OutputImageRegionType;
  typedef typename OutputImageType::PixelType  OutputImagePixelType;

  typedef  std::vector<std::string> FileNamesContainer;

  /** ImageDimension constant */
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  /** Set the vector of strings that contains the file names. Files
   * are processed in sequential order. */
  void SetFileNames (const FileNamesContainer &name)
    {
    if ( m_FileNames != name)
      {
      m_FileNames = name;
      this->Modified();
      }
    }
  const FileNamesContainer & GetFileNames() const
    {
    return m_FileNames;
    }

  /** Prepare the allocation of the output image during the first back
   * propagation of the pipeline. */
  virtual void GenerateOutputInformation(void);

protected:
  ProjectionsReader():m_ImageIO(NULL) {};
  ~ProjectionsReader() {};
  void PrintSelf(std::ostream& os, itk::Indent indent) const;

  /** Does the real work. */
  virtual void GenerateData();

  /** A list of filenames to be processed. */
  FileNamesContainer m_FileNames;

private:
  ProjectionsReader(const Self&); //purposely not implemented
  void operator=(const Self&);    //purposely not implemented

  /** The projections reader which template depends on the scanner.
   * It is not typed because we want to keep the data as on disk.
   * The pointer is stored to reference the filter and avoid its destruction. */
  itk::ProcessObject::Pointer m_RawDataReader;

  /** Conversion from raw to Projections. Is equal to m_RawDataReader
   * if no conversion. Put in a composite filter if more than one operation.*/
  typename itk::ImageSource<TOutputImage>::Pointer m_RawToProjectionsFilter;

  /** Image IO object which is stored to create the pipe only when required */
  itk::ImageIOBase::Pointer m_ImageIO;
};

} //namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkProjectionsReader.txx"
#endif

#endif // __rtkProjectionsReader_h
