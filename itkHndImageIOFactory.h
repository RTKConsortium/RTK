#ifndef ITKHNDIMAGEIOFACTORY_H
#define ITKHNDIMAGEIOFACTORY_H

// clitk include
#include "itkHndImageIO.h"

// itk include
#include "itkImageIOBase.h"
#include "itkObjectFactoryBase.h"
#include "itkVersion.h"

namespace itk
{

/** \class HndImageIOFactory
 *
 * Factory for Hnd files (file format used by Varian for Obi raw data).
 *
 */
class HndImageIOFactory: public itk::ObjectFactoryBase
{
public:
  /** Standard class typedefs. */
  typedef HndImageIOFactory              Self;
  typedef itk::ObjectFactoryBase         Superclass;
  typedef itk::SmartPointer<Self>        Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Class methods used to interface with the registered factories. */
  const char* GetITKSourceVersion(void) const {
    return ITK_SOURCE_VERSION;
  }

  const char* GetDescription(void) const {
    return "His ImageIO Factory, allows the loading of His images into insight";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(HndImageIOFactory, ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void) {
    ObjectFactoryBase::RegisterFactory( Self::New() );
  }

protected:
  HndImageIOFactory();
  ~HndImageIOFactory() {};
  typedef HndImageIOFactory myProductType;
  const myProductType* m_MyProduct;

private:
  HndImageIOFactory(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace

#endif /* end #define ITKHNDIMAGEIOFACTORY_H */

