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
#include "srtkVersion.h"
#include "srtkVersionConfig.h"

namespace
{

std::string MakeExtendedVersionString()
{
  std::ostringstream v;
  v << "SimpleRTK Version: " << rtk::simple::Version::VersionString() << std::endl
    << "Compiled: " << rtk::simple::Version::BuildDate() << std::endl;
  return v.str();
}

static const std::string extendedVersionString = MakeExtendedVersionString();

}

namespace rtk
{
  namespace simple
  {

  unsigned int Version::MajorVersion()
  {
    return SimpleRTK_VERSION_MAJOR;
  }
  unsigned int Version::MinorVersion()
  {
    return SimpleRTK_VERSION_MINOR;
  }
  unsigned int Version::PatchVersion()
  {
    return SimpleRTK_VERSION_PATCH;
  }
  unsigned int Version::TweakVersion()
  {
    return 0;
  }
  const std::string &Version::VersionString()
  {
    static const std::string v( SimpleRTK_VERSION );
    return v;
  }
  const std::string &Version::BuildDate()
  {
    static const std::string v( __DATE__ " " __TIME__ );
    return v;
  }
  const std::string &Version::ExtendedVersionString()
  {
    return extendedVersionString;
  }
  }
}
