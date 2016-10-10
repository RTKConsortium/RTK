/*
 * rtkCalibrationBasedProjectionGeometry.cxx
 *
 *  Created on: 12 nov. 2015
 *      Author: Thibault Notargiacomo
 */


#include "rtkCalibrationBasedProjectionGeometry.h"

//std
#include <cmath>

//ITK
#include <itkVector.h>
#include <itkEuler3DTransform.h>

rtk::CalibrationBasedProjectionGeometry::CalibrationBasedProjectionGeometry()
  : rtk::Reg23ProjectionGeometry()
{
}

rtk::CalibrationBasedProjectionGeometry::~CalibrationBasedProjectionGeometry()
{
}

void rtk::CalibrationBasedProjectionGeometry::AddProjection(
		const double pixelSizeX,
		const double pixelSizeY,
		const Matrix3x4Type &calibrationMatrix,
		const Matrix4x4Type &transformationMatrix )
{
	/*
	 * Comment: use the matrix allowing to transform from a point in
	 * the ITK geometrical framework, into a point in the actual custom system framework
	 */
	Matrix3x4Type currentMatrix( calibrationMatrix.GetVnlMatrix() * transformationMatrix.GetVnlMatrix() );

	/*
	 * Warning: Here, as RTK assumes that the center of the pixel indexed by (0,0) has a floating point index of (0,0)
	 * Although, our matrices are calibrated so that the center of the pixel indexed by (0,0)
	 * has a floating point index of (0.5,0.5), like in cuda and OpencCV
	 * We must add a translation of (-0.5,-0.5) at the left side
	 */

	Matrix3x3Type translation;
	translation.SetIdentity();
	translation(0,2)=-0.5;
	translation(1,2)=-0.5;
	currentMatrix = translation.GetVnlMatrix()*currentMatrix.GetVnlMatrix();

	/*
	 * Now extract parameters thanks to a matrix factorization formula specific to the pinhole model
	 */
	std::vector<double> model;
	_computeParameters( pixelSizeX, pixelSizeY, currentMatrix, model );

	double f = model[0];
	double Tx = model[6];
	double Ty = model[7];
	double Tz = model[8];
	double U0mm = model[1];
	double V0mm = model[2];

	double sid = -Tz;
	double sdd = -f;
	double gantryAngle = -model[4];
	double outOfPlaneAngle = -model[3];
	double inPlaneAngle = -model[5];

	//A little bit more complicated
	double sourceOffsetX = -Tx;
	double sourceOffsetY = -Ty;

	double projOffsetX = sourceOffsetX-U0mm;
	double projOffsetY = sourceOffsetY-V0mm;

	// Add to geometry
	AddProjectionInRadians(
		sid, sdd, gantryAngle,
		projOffsetX, projOffsetY,
		outOfPlaneAngle, inPlaneAngle,
		sourceOffsetX, sourceOffsetY );
}

void rtk::CalibrationBasedProjectionGeometry::_computeParameters(
		const double pixelSizeX,
		const double pixelSizeY,
		const Matrix3x4Type &cMat,
		std::vector<double>& model )
{
	//Initialize enough elements to 0
	model.resize(9,0);
	// u_0
	model[1] = (cMat(0, 0)*cMat(2, 0)) + (cMat(0, 1)*cMat(2, 1)) + (cMat(0, 2)*cMat(2, 2));
	// v_0
	model[2] = (cMat(1, 0)*cMat(2, 0)) + (cMat(1, 1)*cMat(2, 1)) + (cMat(1, 2)*cMat(2, 2));
	// alpha_u
	double aU = sqrt(cMat(0, 0)*cMat(0, 0) + cMat(0, 1)*cMat(0, 1) + cMat(0, 2)*cMat(0, 2) - model[1]*model[1]);
	// alpha_v
	double aV = sqrt(cMat(1, 0)*cMat(1, 0) + cMat(1, 1)*cMat(1, 1) + cMat(1, 2)*cMat(1, 2) - model[2]*model[2]);

	// focal of the system, here we take the mean value
	model[0] = 0.5 * (aU*pixelSizeX + aV*pixelSizeY);

	// Tx
	model[6] = (cMat(0, 3) - model[1]*cMat(2, 3))/aU;
	// Ty
	model[7] = (cMat(1, 3) - model[2]*cMat(2, 3))/aV;
	// Tz
	model[8] = cMat(2, 3);

	Matrix3x3Type rotation;
	for (unsigned int i = 0; i < 3; i++)
	{
		rotation(0,i) = (cMat(0, i)-model[1]*cMat(2, i))/aU;
		rotation(1,i) = (cMat(1, i)-model[2]*cMat(2, i))/aV;
		rotation(2,i) = cMat(2, i);
	}

	//U0 and V0 are expressed in mm
	model[1] *= pixelSizeX;
	model[2] *= pixelSizeY;

	//Declare a 3D euler transform in order to properly extract angles
	typedef itk::Euler3DTransform<double> EulerType;
	EulerType::Pointer euler = EulerType::New();
	euler->SetComputeZYX(false); // ZXY order

	//Extract angle using parent method without orthogonality check, see Reg23ProjectionGeometry.cxx for more
	euler->itk::MatrixOffsetTransformBase<double>::SetMatrix(rotation);
	model[3] = euler->GetAngleX(); // OA
	model[4] = euler->GetAngleY(); // GA
	model[5] = euler->GetAngleZ(); // IA
}
