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
#ifndef __SimpleRTK_h
#define __SimpleRTK_h

#include <stdint.h>

// Utility classes
#include "srtkMacro.h"
#include "srtkDetail.h"
#include "srtkVersion.h"
#include "srtkImage.h"
#include "srtkTransform.h"
#include "srtkThreeDCircularProjectionGeometry.h"
#include "srtkShow.h"

#include "srtkInterpolator.h"
#include "srtkEvent.h"

#include "srtkProcessObject.h"
#include "srtkImageFilter.h"
#include "srtkCommand.h"
#include "srtkFunctionCommand.h"

// IO classes
#include "srtkImageFileReader.h"
#include "srtkImageSeriesReader.h"
#include "srtkProjectionsReader.h"
#include "srtkImageFileWriter.h"
#include "srtkImageSeriesWriter.h"
#include "srtkImportImageFilter.h"
#include "srtkThreeDCircularProjectionGeometryXMLFileReader.h"
#include "srtkThreeDCircularProjectionGeometryXMLFileWriter.h"

#include "srtkHashImageFilter.h"
#include "srtkPixelIDTypeLists.h"
#include "srtkStatisticsImageFilter.h"
#include "srtkCastImageFilter.h"

// These headers are auto-generated
#include "SimpleRTKBasicFiltersGeneratedHeaders.h"
#endif
