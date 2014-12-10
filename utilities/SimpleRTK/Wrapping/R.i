 // R specific swig components
#if SWIGR

// ignore overload methods of int type when there is an enum
%ignore rtk::simple::CastImageFilter::SetOutputPixelType( PixelIDValueType pixelID );
%ignore rtk::simple::GetPixelIDValueAsString( PixelIDValueType type );

%include <std_vector.i>
 // we don't want a class assigned to unsigned char
%typemap(scoerceout) unsigned char,
   unsigned char *,
   unsigned char &
   %{    %}

// Gets rid of the class check for unsigned char function arguments
%typemap("rtype") unsigned char, unsigned char *, unsigned char & "integer";
// and for unsigned int vectors
%typemap("rtype") std::vector<unsigned int>, std::vector<unsigned int> *, std::vector<unsigned int> & "integer";

// some important enumerations don't get evaluate properly. This is a
// hack to fix the problem.

%inline
%{
#include "srtkConditional.h"

  // causes swig problems
  //namespace srtk = rtk::simple;

  rtk::simple::PixelIDValueType RsrtkUInt8 = rtk::simple::srtkUInt8;
  rtk::simple::PixelIDValueType RsrtkInt8  = rtk::simple::srtkInt8;
  rtk::simple::PixelIDValueType RsrtkUInt16 = rtk::simple::srtkUInt16;
  rtk::simple::PixelIDValueType RsrtkInt16  = rtk::simple::srtkInt16;
  rtk::simple::PixelIDValueType RsrtkUInt32 = rtk::simple::srtkUInt32;
  rtk::simple::PixelIDValueType RsrtkInt32  = rtk::simple::srtkInt32;
  rtk::simple::PixelIDValueType RsrtkUInt64 = rtk::simple::srtkUInt64;
  rtk::simple::PixelIDValueType RsrtkInt64  = rtk::simple::srtkInt64;
  rtk::simple::PixelIDValueType RsrtkFloat32 = rtk::simple::srtkFloat32;
  rtk::simple::PixelIDValueType RsrtkFloat64 = rtk::simple::srtkFloat64;

  rtk::simple::PixelIDValueType RsrtkComplexFloat32 = rtk::simple::srtkComplexFloat32;
  rtk::simple::PixelIDValueType RsrtkComplexFloat64 = rtk::simple::srtkComplexFloat64;

  rtk::simple::PixelIDValueType RsrtkVectorUInt8   = rtk::simple::srtkVectorUInt8;
  rtk::simple::PixelIDValueType RsrtkVectorInt8    = rtk::simple::srtkVectorInt8;
  rtk::simple::PixelIDValueType RsrtkVectorUInt16  = rtk::simple::srtkVectorUInt16;
  rtk::simple::PixelIDValueType RsrtkVectorInt16   = rtk::simple::srtkVectorInt16;
  rtk::simple::PixelIDValueType RsrtkVectorUInt32  = rtk::simple::srtkVectorUInt32;
  rtk::simple::PixelIDValueType RsrtkVectorInt32   = rtk::simple::srtkVectorInt32;
  rtk::simple::PixelIDValueType RsrtkVectorUInt64  = rtk::simple::srtkVectorUInt64;
  rtk::simple::PixelIDValueType RsrtkVectorInt64   = rtk::simple::srtkVectorInt64;
  rtk::simple::PixelIDValueType RsrtkVectorFloat32 = rtk::simple::srtkVectorFloat32;
  rtk::simple::PixelIDValueType RsrtkVectorFloat64 = rtk::simple::srtkVectorFloat64;


  rtk::simple::PixelIDValueType RsrtkLabelUInt8  = rtk::simple::srtkLabelUInt8;
  rtk::simple::PixelIDValueType RsrtkLabelUInt16 = rtk::simple::srtkLabelUInt16;
  rtk::simple::PixelIDValueType RsrtkLabelUInt32 = rtk::simple::srtkLabelUInt32;
  rtk::simple::PixelIDValueType RsrtkLabelUInt64 = rtk::simple::srtkLabelUInt64;

  // functions for image content access via bracket operator
  rtk::simple::Image SingleBracketOperator(std::vector<int> xcoord, std::vector<int> ycoord, std::vector<int> zcoord, const rtk::simple::Image src)
    {
      rtk::simple::PixelIDValueType  PID=src.GetPixelIDValue();
      // use 3D coords. They get trimmed appropriately for 2D during access.
      std::vector<unsigned int> scoord(3);
      std::vector<unsigned int> dcoord(3);

      rtk::simple::Image dest;

      if (zcoord.size() > 1)
        {
          dest = rtk::simple::Image(xcoord.size(),ycoord.size(), zcoord.size(),
                                    static_cast<rtk::simple::PixelIDValueEnum>(src.GetPixelIDValue()));
        }
      else
        {
          dest = rtk::simple::Image(xcoord.size(),ycoord.size(),
                                    static_cast<rtk::simple::PixelIDValueEnum>(src.GetPixelIDValue()));

        }
      dest.SetSpacing(src.GetSpacing());

      switch (PID) {
      case rtk::simple::ConditionalValue< rtk::simple::srtkUInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt8, -2 >::Value:
        {
          for (unsigned z = 0, K=0; z < zcoord.size(); z++)
            {
              scoord[2]=static_cast<unsigned int>(zcoord[z]);
              dcoord[2]=K;
              for (unsigned y = 0,J=0; y < ycoord.size(); y++)
                {
                  scoord[1]=static_cast<unsigned int>(ycoord[y]);
                  dcoord[1]=J;
                  for (unsigned x = 0,I=0; x < xcoord.size(); x++)
                    {
                      scoord[0]=static_cast<unsigned int>(xcoord[x]);
                      dcoord[0]=I;
                      dest.SetPixelAsUInt8(dcoord, src.GetPixelAsUInt8(scoord));
                      I++;
                    }
                  J++;
                }
              K++;
            }
          return(dest);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkInt8, -3 >::Value:
        {
          for (unsigned z = 0, K=0; z < zcoord.size(); z++)
            {
              scoord[2]=static_cast<unsigned int>(zcoord[z]);
              dcoord[2]=K;
              for (unsigned y = 0,J=0; y < ycoord.size(); y++)
                {
                  scoord[1]=static_cast<unsigned int>(ycoord[y]);
                  dcoord[1]=J;
                  for (unsigned x = 0,I=0; x < xcoord.size(); x++)
                    {
                      scoord[0]=static_cast<unsigned int>(xcoord[x]);
                      dcoord[0]=I;
                      dest.SetPixelAsInt8(dcoord, src.GetPixelAsInt8(scoord));
                      I++;
                    }
                  J++;
                }
              K++;
            }
          return(dest);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkUInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt16, -4 >::Value:
        {
          for (unsigned z = 0, K=0; z < zcoord.size(); z++)
            {
              scoord[2]=static_cast<unsigned int>(zcoord[z]);
              dcoord[2]=K;
              for (unsigned y = 0,J=0; y < ycoord.size(); y++)
                {
                  scoord[1]=static_cast<unsigned int>(ycoord[y]);
                  dcoord[1]=J;
                  for (unsigned x = 0,I=0; x < xcoord.size(); x++)
                    {
                      scoord[0]=static_cast<unsigned int>(xcoord[x]);
                      dcoord[0]=I;
                      dest.SetPixelAsUInt16(dcoord, src.GetPixelAsUInt16(scoord));
                      I++;
                    }
                  J++;
                }
              K++;
            }
          return(dest);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkInt16, -5 >::Value:
        {
          for (unsigned z = 0, K=0; z < zcoord.size(); z++)
            {
              scoord[2]=static_cast<unsigned int>(zcoord[z]);
              dcoord[2]=K;
              for (unsigned y = 0,J=0; y < ycoord.size(); y++)
                {
                  scoord[1]=static_cast<unsigned int>(ycoord[y]);
                  dcoord[1]=J;
                  for (unsigned x = 0,I=0; x < xcoord.size(); x++)
                    {
                      scoord[0]=static_cast<unsigned int>(xcoord[x]);
                      dcoord[0]=I;
                      dest.SetPixelAsInt16(dcoord, src.GetPixelAsInt16(scoord));
                      I++;
                    }
                  J++;
                }
              K++;
            }
          return(dest);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkUInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt32, -6 >::Value:
        {
          for (unsigned z = 0, K=0; z < zcoord.size(); z++)
            {
              scoord[2]=static_cast<unsigned int>(zcoord[z]);
              dcoord[2]=K;
              for (unsigned y = 0,J=0; y < ycoord.size(); y++)
                {
                  scoord[1]=static_cast<unsigned int>(ycoord[y]);
                  dcoord[1]=J;
                  for (unsigned x = 0,I=0; x < xcoord.size(); x++)
                    {
                      scoord[0]=static_cast<unsigned int>(xcoord[x]);
                      dcoord[0]=I;
                      dest.SetPixelAsUInt32(dcoord, src.GetPixelAsUInt32(scoord));
                      I++;
                    }
                  J++;
                }
              K++;
            }
          return(dest);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkInt32, -7 >::Value:
        {
          for (unsigned z = 0, K=0; z < zcoord.size(); z++)
            {
              scoord[2]=static_cast<unsigned int>(zcoord[z]);
              dcoord[2]=K;
              for (unsigned y = 0,J=0; y < ycoord.size(); y++)
                {
                  scoord[1]=static_cast<unsigned int>(ycoord[y]);
                  dcoord[1]=J;
                  for (unsigned x = 0,I=0; x < xcoord.size(); x++)
                    {
                      scoord[0]=static_cast<unsigned int>(xcoord[x]);
                      dcoord[0]=I;
                      dest.SetPixelAsInt32(dcoord, src.GetPixelAsInt32(scoord));
                      I++;
                    }
                  J++;
                }
              K++;
            }
          return(dest);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkUInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt64, -8 >::Value:
        {
          for (unsigned z = 0, K=0; z < zcoord.size(); z++)
            {
              scoord[2]=static_cast<unsigned int>(zcoord[z]);
              dcoord[2]=K;
              for (unsigned y = 0,J=0; y < ycoord.size(); y++)
                {
                  scoord[1]=static_cast<unsigned int>(ycoord[y]);
                  dcoord[1]=J;
                  for (unsigned x = 0,I=0; x < xcoord.size(); x++)
                    {
                      scoord[0]=static_cast<unsigned int>(xcoord[x]);
                      dcoord[0]=I;
                      dest.SetPixelAsUInt64(dcoord, src.GetPixelAsUInt64(scoord));
                      I++;
                    }
                  J++;
                }
              K++;
            }
          return(dest);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkInt64, -9 >::Value:
        {
          for (unsigned z = 0, K=0; z < zcoord.size(); z++)
            {
              scoord[2]=static_cast<unsigned int>(zcoord[z]);
              dcoord[2]=K;
              for (unsigned y = 0,J=0; y < ycoord.size(); y++)
                {
                  scoord[1]=static_cast<unsigned int>(ycoord[y]);
                  dcoord[1]=J;
                  for (unsigned x = 0,I=0; x < xcoord.size(); x++)
                    {
                      scoord[0]=static_cast<unsigned int>(xcoord[x]);
                      dcoord[0]=I;
                      dest.SetPixelAsInt64(dcoord, src.GetPixelAsInt64(scoord));
                      I++;
                    }
                  J++;
                }
              K++;
            }
          return(dest);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkFloat32 != rtk::simple::srtkUnknown, rtk::simple::srtkFloat32, -10 >::Value:

        {
          for (unsigned z = 0, K=0; z < zcoord.size(); z++)
            {
              scoord[2]=static_cast<unsigned int>(zcoord[z]);
              dcoord[2]=K;
              for (unsigned y = 0,J=0; y < ycoord.size(); y++)
                {
                  scoord[1]=static_cast<unsigned int>(ycoord[y]);
                  dcoord[1]=J;
                  for (unsigned x = 0,I=0; x < xcoord.size(); x++)
                    {
                      scoord[0]=static_cast<unsigned int>(xcoord[x]);
                      dcoord[0]=I;
                      dest.SetPixelAsFloat(dcoord, src.GetPixelAsFloat(scoord));
                      I++;
                    }
                  J++;
                }
              K++;
            }
          return(dest);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkFloat64 != rtk::simple::srtkUnknown, rtk::simple::srtkFloat64, -11 >::Value:
        {
          for (unsigned z = 0, K=0; z < zcoord.size(); z++)
            {
              scoord[2]=static_cast<unsigned int>(zcoord[z]);
              dcoord[2]=K;
              for (unsigned y = 0,J=0; y < ycoord.size(); y++)
                {
                  scoord[1]=static_cast<unsigned int>(ycoord[y]);
                  dcoord[1]=J;
                  for (unsigned x = 0,I=0; x < xcoord.size(); x++)
                    {
                      scoord[0]=static_cast<unsigned int>(xcoord[x]);
                      dcoord[0]=I;
                      dest.SetPixelAsDouble(dcoord, src.GetPixelAsDouble(scoord));
                      I++;
                    }
                  J++;
                }
              K++;
            }
          return(dest);
        }
      case rtk::simple::ConditionalValue< rtk::simple::srtkComplexFloat32 != rtk::simple::srtkUnknown, rtk::simple::srtkComplexFloat32, -12 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkComplexFloat64 != rtk::simple::srtkUnknown, rtk::simple::srtkComplexFloat64, -13 >::Value:
        {
          char error_msg[1024];
          snprintf( error_msg, 1024, "Exception thrown SingleBracketOperator : complex floating types not supported");
          Rprintf(error_msg);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt8, -14 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt8, -15 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt16, -16 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt16, -17 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt32, -18 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt32, -19 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt64, -20 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt64, -21 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorFloat32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorFloat32, -22 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorFloat64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorFloat64, -23 >::Value:
        {
        char error_msg[1024];
        snprintf( error_msg, 1024, "Images of Vector Pixel types currently are not supported." );
        Rprintf(error_msg);
        }
        break;
      default:
        char error_msg[1024];
        snprintf( error_msg, 1024, "Exception thrown SingleBrackeOperator : unsupported pixel type: %d", PID );
        Rprintf(error_msg);
      }
      // return something to keep R happy.
      return(rtk::simple::Image(0,0,0, rtk::simple::srtkUInt8));
    }

  SEXP ImAsArray(rtk::simple::Image src)
  {
    // tricky to make this efficient with memory and fast.
    // Ideally we want multithreaded casting directly to the
    // R array. We could use a Cast filter and then a memory copy,
    // obviously producing a redundant copy. If we do a direct cast,
    // then we're probably not multi-threaded.
    // Lets be slow but memory efficient.

    std::vector<unsigned int> sz = src.GetSize();
    rtk::simple::PixelIDValueType  PID=src.GetPixelIDValue();
    SEXP res = 0;
    double *dans=0;
    int *ians=0;
    unsigned pixcount=src.GetNumberOfComponentsPerPixel();
    for (unsigned k = 0; k < sz.size();k++)
      {
        pixcount *= sz[k];
      }
    switch (PID)
      {
      case rtk::simple::srtkUnknown:
        {
          char error_msg[1024];
          snprintf( error_msg, 1024, "Exception thrown ImAsArray : unkown pixel type");
          Rprintf(error_msg);
          return(res);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkInt8, -3 >::Value:
case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt8, -15 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkUInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt8, -2 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt8, -14 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkInt16, -5 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt16, -17 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkUInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt16, -4 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt16, -16 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkInt32, -7 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt32, -19 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkUInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt32, -6 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt32, -18 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkUInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt64, -8 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt64, -20 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkInt64, -9 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt64, -21 >::Value:
        {
          // allocate an integer array
          PROTECT(res = Rf_allocVector(INTSXP, pixcount));
          ians = INTEGER_POINTER(res);
        }
        break;
      default:
        {
          // allocate double array
          PROTECT(res = Rf_allocVector(REALSXP, pixcount));
          dans = NUMERIC_POINTER(res);
        }
      }

    switch (PID)
      {
      case rtk::simple::ConditionalValue< rtk::simple::srtkUInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt8, -2 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt8, -14 >::Value:
        {
        uint8_t * buff = src.GetBufferAsUInt8();
        std::copy(buff,buff + pixcount,ians);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkInt8, -3 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt8, -15 >::Value:
        {
        int8_t * buff = src.GetBufferAsInt8();
        std::copy(buff,buff + pixcount,ians);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkUInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt16, -4 >::Value:
case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt16, -16 >::Value:
        {
        uint16_t * buff = src.GetBufferAsUInt16();
        std::copy(buff,buff + pixcount,ians);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkInt16, -5 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt16, -17 >::Value:
        {
        int16_t * buff = src.GetBufferAsInt16();
        std::copy(buff,buff + pixcount,ians);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkUInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt32, -6 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt32, -18 >::Value:
        {
        uint32_t * buff = src.GetBufferAsUInt32();
        std::copy(buff,buff + pixcount,ians);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkInt32, -7 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt32, -19 >::Value:
        {
        int32_t * buff = src.GetBufferAsInt32();
        std::copy(buff,buff + pixcount,ians);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkUInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt64, -8 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt64, -20 >::Value:

        {
        uint64_t * buff = src.GetBufferAsUInt64();
        std::copy(buff,buff + pixcount,ians);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkInt64, -9 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt64, -21 >::Value:
        {
        int64_t * buff = src.GetBufferAsInt64();
        std::copy(buff,buff + pixcount,ians);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkFloat32 != rtk::simple::srtkUnknown, rtk::simple::srtkFloat32, -10 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorFloat32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorFloat32, -22 >::Value:
        {
        float * buff = src.GetBufferAsFloat();
        std::copy(buff,buff + pixcount,dans);
        }
        break;
      case rtk::simple::ConditionalValue< rtk::simple::srtkFloat64 != rtk::simple::srtkUnknown, rtk::simple::srtkFloat64, -11 >::Value:
      case rtk::simple::ConditionalValue< rtk::simple::srtkVectorFloat64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorFloat64, -23 >::Value:
        {
        double * buff = src.GetBufferAsDouble();
        std::copy(buff,buff + pixcount,dans);
        }
        break;
      default:
        char error_msg[1024];
        snprintf( error_msg, 1024, "Exception thrown ImAsArray : unsupported pixel type: %d", PID );
        Rprintf(error_msg);
      }
    UNPROTECT(1);
    return(res);
  }

#include "srtkImportImageFilter.h"

rtk::simple::Image ArrayAsIm(SEXP arr,
                             std::vector<unsigned int> size,
                             std::vector<double> spacing,
                             std::vector<double> origin)
    {
      // can't work out how to get the array size in C
      rtk::simple::ImportImageFilter importer;
      importer.SetSpacing( spacing );
      importer.SetOrigin( origin );
      importer.SetSize( size );
      if (Rf_isReal(arr))
        {
          importer.SetBufferAsDouble(NUMERIC_POINTER(arr));
        }
      else if (Rf_isInteger(arr) || Rf_isLogical(arr))
        {
          importer.SetBufferAsInt32(INTEGER_POINTER(arr));
        }
      else
        {
          char error_msg[1024];
          snprintf( error_msg, 1024, "Exception thrown ArrayAsIm : unsupported array type");
          Rprintf(error_msg);
        }
      rtk::simple::Image res = importer.Execute();
      return(res);
    }

  %}

//#define %rcode %insert("sinit")

%Rruntime %{

  setMethod('show', '_p_itk__simple__Image', function(object) Show(object))
  setMethod('print', '_p_itk__simple__Image', function(x, ...)cat(x$ToString()))

  setMethod('print', 'C++Reference', function(x, ...)cat(x$ToString()))
  setMethod('show', 'C++Reference', function(object)cat(object$ToString()))

    %}
#endif
