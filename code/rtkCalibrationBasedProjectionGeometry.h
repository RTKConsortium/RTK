/*
 * rtkCalibrationBasedProjectionGeometry.h
 *
 *  Created on: 12 nov. 2015
 *      Author: Thibault Notargiacomo
 */

#ifndef CODE_RTKCALIBRATIONBASEDPROJECTIONGEOMETRY_H_
#define CODE_RTKCALIBRATIONBASEDPROJECTIONGEOMETRY_H_


//RTK
#include "rtkReg23ProjectionGeometry.h"

//STL
#include <vector>

namespace rtk
{

/** \class CalibrationBasedProjectionGeometry
 * \brief {A simple utility class which allows to import 3*4 projection matrices coming from a "blind" calibration into RTK.}
 *
 * This class features a method that allows to import a 3*4 projection matrix, we will call M into a proper RTK geometry.
 * We assume here that M allows to transform from a 3D+1 geometry world to a 2D homogeneous pixel index world.
 * This kind of matrix is quite common as it can be easily obtained by solving a simple linear least square :
 * See for instance https://siddhantahuja.wordpress.com/2010/02/20/570/
 *
 * @warning {It is important that the user provide X and Y pixel size informations, X being related to the first coordinate
 * of the homogeneous coordinate system obtained after a projection by M, a Y the second. }
 *
 * @warning {The detector offset and direction is fully modeled by the resultant
 * RTK geometry entry. This means that the projection stack input into back-
 * projection filters or forward-projectors is expected to have zero-origin and
 * identity-direction!}
 *
 * @see rtk::Reg23ProjectionGeometry
 *
 * @author Thibault Notargiacomo
 * @version 1.0
 */
class RTK_EXPORT CalibrationBasedProjectionGeometry :
    public rtk::Reg23ProjectionGeometry
{
public:
  /** General typedefs **/
  typedef CalibrationBasedProjectionGeometry    Self;
  typedef rtk::Reg23ProjectionGeometry			Superclass;
  typedef itk::SmartPointer<Self>               Pointer;
  typedef itk::SmartPointer<const Self>         ConstPointer;
  typedef itk::Point<double, 3>                 PointType;
  typedef itk::Matrix<double, 3, 3>             Matrix3x3Type;
  typedef itk::Matrix<double, 3, 4>             Matrix3x4Type;
  typedef itk::Matrix<double, 4, 4>             Matrix4x4Type;


  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /**
   * @brief Add a REG23-based geometry set to the RTK projections list.
   * @param sourcePosition absolute position of the point source S in WCS
   * @param detectorPosition absolute position of the detector origin R in WCS
   * @param detectorRowVector absolute direction vector indicating the
   * orientation of the detector's rows r (sometimes referred to as v1)
   * @param detectorColumnVector absolute direction vector indicating the
   * orientation of the detector's columns c (sometimes referred to as v2)
   * @return TRUE if the projection could be added to the RTK projections list
   */
  void AddProjection(	const double pixelSizeX,
		  	  	  	  	const double pixelSizeY,
						const Matrix3x4Type &calibrationMatrix,
						const Matrix4x4Type &transformationMatrix ); //TODO TN: set to Identity by default

protected:

  /**
   * Uses a custom matrix factorization technic for pinhole camera model parameter extraction
   */
  void _computeParameters(
		const double pixelSizeX,
		const double pixelSizeY,
  		const Matrix3x4Type &calibrationMatrix,
  		std::vector<double>& param );

protected:
  /** Standard constructor. **/
  CalibrationBasedProjectionGeometry();
  /** Destructor. **/
  virtual ~CalibrationBasedProjectionGeometry();

private:
  /** Purposely not implemented. **/
  CalibrationBasedProjectionGeometry(const Self&);
  /** Purposely not implemented. **/
  void operator=(const Self&);

};

}



#endif /* CODE_RTKCALIBRATIONBASEDPROJECTIONGEOMETRY_H_ */
