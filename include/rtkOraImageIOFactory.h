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

#ifndef rtkOraImageIOFactory_h
#define rtkOraImageIOFactory_h

#include "RTKExport.h"
#include "rtkOraImageIO.h"

// itk include
#include <itkImageIOBase.h>
#include <itkObjectFactoryBase.h>
#include <itkVersion.h>

namespace rtk
{

/** \class OraImageIOFactory
 * \brief ITK factory for Ora file I/O.
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
class RTK_EXPORT OraImageIOFactory : public itk::ObjectFactoryBase
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(OraImageIOFactory);
#else
  ITK_DISALLOW_COPY_AND_MOVE(OraImageIOFactory);
#endif

  /** Standard class type alias. */
  using Self = OraImageIOFactory;
  using Superclass = itk::ObjectFactoryBase;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Class methods used to interface with the registered factories. */
  const char *
  GetITKSourceVersion() const override
  {
    return ITK_SOURCE_VERSION;
  }

  const char *
  GetDescription() const override
  {
    return "Ora ImageIO Factory, allows the loading of Ora images into insight";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(OraImageIOFactory, itk::ObjectFactoryBase);

  /** Register one factory of this type  */
  static void
  RegisterOneFactory()
  {
    ObjectFactoryBase::RegisterFactory(Self::New());
  }

protected:
  OraImageIOFactory();
  ~OraImageIOFactory() override = default;
  using myProductType = OraImageIOFactory;
  const myProductType * m_MyProduct;
};

} // namespace rtk

#endif /* end #define rtkOraImageIOFactory_h */
