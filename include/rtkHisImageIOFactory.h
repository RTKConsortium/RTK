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

#ifndef rtkHisImageIOFactory_h
#define rtkHisImageIOFactory_h

#include "RTKExport.h"
#include "rtkHisImageIO.h"
#include "rtkMacro.h"

// itk include
#include <itkImageIOBase.h>
#include <itkObjectFactoryBase.h>
#include <itkVersion.h>

namespace rtk
{

/** \class HisImageIOFactory
 * \brief ITK factory for His file I/O.
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
class RTK_EXPORT HisImageIOFactory : public itk::ObjectFactoryBase
{
public:
  /** Standard class type alias. */
  using Self = HisImageIOFactory;
  using Superclass = itk::ObjectFactoryBase;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Class methods used to interface with the registered factories. */
  const char* GetITKSourceVersion(void) const override {
    return ITK_SOURCE_VERSION;
  }

  const char* GetDescription(void) const override {
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
  ~HisImageIOFactory() override = default;
  using myProductType = HisImageIOFactory;
  const myProductType* m_MyProduct;

private:
  HisImageIOFactory(const Self&); //purposely not implemented
  void operator=(const Self&);    //purposely not implemented

};

} // end namespace

#endif /* end #define rtkHisImageIOFactory_h */
