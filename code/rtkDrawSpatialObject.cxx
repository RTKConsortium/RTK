#include "rtkDrawSpatialObject.h"





rtk::DrawCylinderSpatialObject::DrawCylinderSpatialObject()
{

}


bool rtk::DrawSpatialObject::IsInside(const rtk::DrawSpatialObject::PointType& point) const
{
 return true;

}



bool rtk::DrawCylinderSpatialObject::IsInside(const rtk::DrawCylinderSpatialObject::PointType& point) const
{
      double QuadricEllip = sqpFunctor->GetA()*point[0]*point[0]   +
                          sqpFunctor->GetB()*point[1]*point[1]   +
                          sqpFunctor->GetC()*point[2]*point[2]   +
                          sqpFunctor->GetD()*point[0]*point[1]   +
                          sqpFunctor->GetE()*point[0]*point[2]   +
                          sqpFunctor->GetF()*point[1]*point[2]   +
                          sqpFunctor->GetG()*point[0] + sqpFunctor->GetH()*point[1] +
                          sqpFunctor->GetI()*point[2] + sqpFunctor->GetJ();
			  
			  
   if(QuadricEllip<0)
      return true;    
   return false;    
  
  
}


