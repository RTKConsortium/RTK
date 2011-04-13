#ifndef __itkProjectionsReader_h
#define __itkProjectionsReader_h

// ITK
#include <itkImageSource.h>
#include <itkImageIOFactory.h>

// Standard lib
#include <vector>
#include <string>

namespace itk
{

/** \class ProjectionsReader
 *
 * This is the universal projections reader of rtk (raw data converted to
 * understandable values, e.g. attenuation).
 *
 */
template <class TOutputImage>
class ITK_EXPORT ProjectionsReader : public ImageSource<TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ProjectionsReader         Self;
  typedef ImageSource<TOutputImage> Superclass;
  typedef SmartPointer<Self>        Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ProjectionsReader, ImageSource);

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
  void PrintSelf(std::ostream& os, Indent indent) const;

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
  ProcessObject::Pointer m_RawDataReader;

  /** Conversion from raw to Projections. Is equal to m_RawDataReader
   * if no conversion. Put in a composite filter if more than one operation.*/
  typename ImageSource<TOutputImage>::Pointer m_RawToProjectionsFilter;

  /** Image IO object which is stored to create the pipe only when required */
  itk::ImageIOBase::Pointer m_ImageIO;
};

} //namespace ITK

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkProjectionsReader.txx"
#endif

#endif // __itkProjectionsReader_h
