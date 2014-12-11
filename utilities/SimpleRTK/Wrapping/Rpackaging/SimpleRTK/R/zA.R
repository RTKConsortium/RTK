.onLoad <- function(lib,pkg) {
  library.dynam("SimpleRTK",pkg,lib)

  defineEnumeration('_itk__simple__PixelIDValueEnum',
                    .values = c(
                      'srtkUnknown' = -1,
                      'srtkUInt8' = RsrtkUInt8(),
                      'srtkInt8' = RsrtkInt8(),
                      'srtkUInt16' = RsrtkUInt16(),
                      'srtkInt16' = RsrtkInt16(),
                      'srtkUInt32' = RsrtkUInt32(),
                      'srtkInt32' = RsrtkInt32(),
                      'srtkUInt64' = RsrtkUInt64(),
                      'srtkInt64' = RsrtkInt64(),
                      'srtkFloat32' = RsrtkFloat32(),
                      'srtkFloat64' = RsrtkFloat64(),
                      'srtkComplexFloat32' = RsrtkComplexFloat32(),
                      'srtkComplexFloat64' = RsrtkComplexFloat64(),
                      'srtkVectorUInt8' = RsrtkVectorUInt8(),
                      'srtkVectorInt8' = RsrtkVectorInt8(),
                      'srtkVectorUInt16' = RsrtkVectorUInt16(),
                      'srtkVectorInt16' = RsrtkVectorInt16(),
                      'srtkVectorUInt32' = RsrtkVectorUInt32(),
                      'srtkVectorInt32' = RsrtkVectorInt32(),
                      'srtkVectorUInt64' = RsrtkVectorUInt64(),
                      'srtkVectorInt64' = RsrtkVectorInt64(),
                      'srtkVectorFloat32' = RsrtkVectorFloat32(),
                      'srtkVectorFloat64' = RsrtkVectorFloat64(),
                      'srtkLabelUInt8' = RsrtkLabelUInt8(),
                      'srtkLabelUInt16' = RsrtkLabelUInt16(),
                      'srtkLabelUInt32' = RsrtkLabelUInt32(),
                      'srtkLabelUInt64' = RsrtkLabelUInt64()
                      ))

  defineEnumeration('_itk__simple__PixelGetEnum',
                    .values = c(
                      'srtkUnknown' = -1,
                      'Image_GetPixelAsUInt8' = RsrtkUInt8(),
                      'Image_GetPixelAsInt8' = RsrtkInt8(),
                      'Image_GetPixelAsiUInt16' = RsrtkUInt16(),
                      'Image_GetPixelAsInt16' = RsrtkInt16(),
                      'Image_GetPixelAsUInt32' = RsrtkUInt32(),
                      'Image_GetPixelAsInt32' = RsrtkInt32(),
                      'Image_GetPixelAsUInt64' = RsrtkUInt64(),
                      'Image_GetPixelAsInt64' = RsrtkInt64(),
                      'Image_GetPixelAsFloat' = RsrtkFloat32(),
                      'Image_GetPixelAsFloat' = RsrtkFloat64()
                      ))

  defineEnumeration('_itk__simple__PixelSetEnum',
                    .values = c(
                      'srtkUnknown' = -1,
                      'Image_SetPixelAsUInt8' = RsrtkUInt8(),
                      'Image_SetPixelAsInt8' = RsrtkInt8(),
                      'Image_SetPixelAsiUInt16' = RsrtkUInt16(),
                      'Image_SetPixelAsInt16' = RsrtkInt16(),
                      'Image_SetPixelAsUInt32' = RsrtkUInt32(),
                      'Image_SetPixelAsInt32' = RsrtkInt32(),
                      'Image_SetPixelAsUInt64' = RsrtkUInt64(),
                      'Image_SetPixelAsInt64' = RsrtkInt64(),
                      'Image_SetPixelAsFloat' = RsrtkFloat32(),
                      'Image_SetPixelAsFloat' = RsrtkFloat64()
                      ))

}

                                        # experimental bracket operator for images
setMethod('[', "_p_itk__simple__Image",
          function(x,i, j, k, drop=TRUE) {
                                        # check to see whether this is returning a single number or an image
            m <- sys.call()

            imdim <- Image_GetDimension(x)
            if ((length(m)-2) < imdim)
              {
                stop("Image has more dimensions")
              }
            imsize <- rep(1, 5)
            imsize[1:imdim] <- Image_GetSize(x)

            if (missing(i)) {
              i <- 1:imsize[1]
            } else {
              i <- (1:imsize[1])[i]
            }

            if (missing(j)) {
              j <- 1:imsize[2]
            } else {
              j <- (1:imsize[2])[j]
            }
            if (missing(k)) {
              k <- 1:imsize[3]
            } else {
              k <- (1:imsize[3])[k]
            }


            if (any(is.na(c(i,j,k)))) {
              stop("Indexes out of range")
            }
            i <- i - 1
            j <- j - 1
            k <- k - 1
            if ((length(i) == 1) & (length(j) == 1) & (length(k) == 1) ) {
              ## return a single point
              pixtype <- x$GetPixelIDValue()
              afName <- enumFromInteger(pixtype, "_itk__simple__PixelGetEnum")
              aF <- get(afName)
              if (!is.null(aF)) {
                ## need to check whether we are using R or C indexing.
                return(aF(x, c(i, j,k)))
              }
            } else {
              ## construct and return an image
              pixtype <- x$GetPixelIDValue()
              resIm <- SingleBracketOperator(i,j,k,x)
              return(resIm);
            }

          }

          )

setMethod('as.array', "_p_itk__simple__Image",
          function(x, drop=TRUE) {
            sz <- x$GetSize()
            if (.hasSlot(x, "ref")) x = slot(x,"ref")
            ans = .Call("R_swig_ImAsArray", x, FALSE, PACKAGE = "SimpleRTK")
            dim(ans) <- sz
            if (drop)
              return(drop(ans))
            return(ans)

            }
          )

as.image <- function(arr, spacing=rep(1, length(dim(arr))),
                     origin=rep(0,length(dim(arr))))
  {
    size <- dim(arr)
    return(ArrayAsIm(arr, size, spacing,origin))
  }
