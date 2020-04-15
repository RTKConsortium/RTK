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

#ifndef rtkHncImageIOFactory_h
#define rtkHncImageIOFactory_h

#include "RTKExport.h"
#include "rtkHncImageIO.h"

// itk include
#include <itkImageIOBase.h>
#include <itkObjectFactoryBase.h>
#include <itkVersion.h>

namespace rtk
{

/** \class HncImageIOFactory
 *
 * Factory for Hnc files (file format used by Varian for Obi raw data).
 *
 * \author Geoff Hugo, VCU
 *
 * \ingroup RTK IOFilters
 */
class HncImageIOFactory : public itk::ObjectFactoryBase
{
public:
  /** Standard class type alias. */
  using Self = HncImageIOFactory;
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
    return "Hnc ImageIO Factory, allows the loading of Hnc images into insight";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(HncImageIOFactory, itk::ObjectFactoryBase);

  /** Register one factory of this type  */
  static void
  RegisterOneFactory()
  {
    ObjectFactoryBase::RegisterFactory(Self::New());
  }

  HncImageIOFactory(const Self &) = delete;
  void
  operator=(const Self &) = delete;

protected:
  HncImageIOFactory();
  ~HncImageIOFactory() override = default;
  using myProductType = HncImageIOFactory;
  const myProductType * m_MyProduct;
};

} // namespace rtk

#endif // rtkHncImageIOFactory_h
