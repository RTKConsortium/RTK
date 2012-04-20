#ifndef __rtkHisImageIOFactory_h
#define __rtkHisImageIOFactory_h

#include "rtkHisImageIO.h"

// itk include
#include <itkImageIOBase.h>
#include <itkObjectFactoryBase.h>
#include <itkVersion.h>

namespace rtk
{

//====================================================================
// Factory for reading His Image file format
class HisImageIOFactory : public itk::ObjectFactoryBase
{
public:
  /** Standard class typedefs. */
  typedef HisImageIOFactory             Self;
  typedef itk::ObjectFactoryBase        Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

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
  itkTypeMacro(HisImageIOFactory, itk::ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void) {
    ObjectFactoryBase::RegisterFactory( Self::New() );
  }

protected:
  HisImageIOFactory();
  ~HisImageIOFactory() {};
  typedef HisImageIOFactory myProductType;
  const myProductType* m_MyProduct;

private:
  HisImageIOFactory(const Self&); //purposely not implemented
  void operator=(const Self&);    //purposely not implemented

};

} // end namespace

#endif /* end #define __rtkHisImageIOFactory_h */
