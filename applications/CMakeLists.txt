FIND_PACKAGE(Gengetopt)

#=========================================================
# Executables
ADD_SUBDIRECTORY(rtkprojections)
ADD_SUBDIRECTORY(rtkparkershortscanweighting)
ADD_SUBDIRECTORY(rtkwangdisplaceddetectorweighting)
ADD_SUBDIRECTORY(rtkfdktwodweights)
ADD_SUBDIRECTORY(rtkramp)
ADD_SUBDIRECTORY(rtkfdk)
ADD_SUBDIRECTORY(rtkinlinefdk)
ADD_SUBDIRECTORY(rtkamsterdamshroud)
ADD_SUBDIRECTORY(rtkforwardprojections)
ADD_SUBDIRECTORY(rtkbackprojections)
ADD_SUBDIRECTORY(rtkrayboxintersection)
ADD_SUBDIRECTORY(rtksart)
ADD_SUBDIRECTORY(rtkrayquadricintersection)
ADD_SUBDIRECTORY(rtkprojectgeometricphantom)
ADD_SUBDIRECTORY(rtkdrawgeometricphantom)
ADD_SUBDIRECTORY(rtkprojectshepploganphantom)
ADD_SUBDIRECTORY(rtkfieldofview)
ADD_SUBDIRECTORY(rtksubselect)
ADD_SUBDIRECTORY(rtkdrawshepploganphantom)
ADD_SUBDIRECTORY(rtkextractshroudsignal)


#All the executables below are meant to create RTK ThreeDCircularProjectionGeometry files
ADD_SUBDIRECTORY(rtkvarianobigeometry)
ADD_SUBDIRECTORY(rtksimulatedgeometry)
ADD_SUBDIRECTORY(rtkelektasynergygeometry)
ADD_SUBDIRECTORY(rtkdigisensgeometry)
#=========================================================

