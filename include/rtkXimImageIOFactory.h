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

//#ifndef ITKHNDIMAGEIOFACTORY_H
//#define ITKHNDIMAGEIOFACTORY_H

#ifndef rtkXimImageIOFactory_h
#define rtkXimImageIOFactory_h

#include "RTKExport.h"
#include "rtkXimImageIO.h"
#include "rtkMacro.h"

// itk include
#include <itkImageIOBase.h>
#include <itkObjectFactoryBase.h>
#include <itkVersion.h>

namespace rtk
{

/** \class XimImageIOFactory
 * \brief ITK factory for Xim file I/O.
 *
 * \author Simon Rit & Andreas Gravgaard Andersen
 *
 * \ingroup RTK
 */
class RTK_EXPORT XimImageIOFactory : public itk::ObjectFactoryBase
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(XimImageIOFactory);

  /** Standard class type alias. */
  using Self = XimImageIOFactory;
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
    return "Xim ImageIO Factory, allows the loading of Xim images into insight";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(XimImageIOFactory, itk::ObjectFactoryBase);

  /** Register one factory of this type  */
  static void
  RegisterOneFactory()
  {
    ObjectFactoryBase::RegisterFactory(Self::New());
  }

protected:
  XimImageIOFactory();
  ~XimImageIOFactory() override = default;
  using myProductType = XimImageIOFactory;
  const myProductType * m_MyProduct;
};

} // namespace rtk

#endif // __rtkXimImageIOFactory_h
