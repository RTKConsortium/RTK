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

#ifndef rtkImagXImageIOFactory_h
#define rtkImagXImageIOFactory_h

#include "rtkWin32Header.h"
#include "rtkImagXImageIO.h"
#include <itkImageIOBase.h>
#include <itkObjectFactoryBase.h>
#include <itkVersion.h>

namespace rtk
{

/** \class ImagXImageIOFactory
 *
 * TODO
 *
 */
class RTK_EXPORT ImagXImageIOFactory : public itk::ObjectFactoryBase
{
public:
  /** Standard class typedefs. */
  typedef ImagXImageIOFactory           Self;
  typedef itk::ObjectFactoryBase        Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Class methods used to interface with the registered factories. */
  const char* GetITKSourceVersion(void) const ITK_OVERRIDE {
    return ITK_SOURCE_VERSION;
  }

  const char* GetDescription(void) const ITK_OVERRIDE {
    return "ImagX ImageIO Factory, allows the loading of ImagX images into insight";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImagXImageIOFactory, ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void) {
    ObjectFactoryBase::RegisterFactory( Self::New() );
  }

protected:
  ImagXImageIOFactory();
  ~ImagXImageIOFactory() {}
  typedef ImagXImageIOFactory myProductType;
  const myProductType* m_MyProduct;
private:
  ImagXImageIOFactory(const Self&); //purposely not implemented
  void operator=(const Self&);      //purposely not implemented

};

} // end namespace

#endif
