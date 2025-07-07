import sys
import importlib

itk_module = sys.modules["itk"]
rtk_module = getattr(itk_module, "RTK")

# Import RTK submodules
rtk_submodules = [
    "itk.rtkversion",
    "itk.rtkargumentparser",
    "itk.rtkinputprojections_group",
    "itk.rtk3Doutputimage_group",
    "itk.rtkprojectors_group",
    "itk.rtkiterations_group",
    "itk.rtkExtras",
]
for mod_name in rtk_submodules:
    mod = importlib.import_module(mod_name)
    for a in dir(mod):
        if a[0] != "_":
            setattr(rtk_module, a, getattr(mod, a))

# Application modules
_app_modules = [
    "rtkadmmtotalvariation",
    "rtkamsterdamshroud",
    "rtkbackprojections",
    "rtkbioscangeometry",
    "rtkcheckimagequality",
    "rtkconjugategradient",
    "rtkdigisensgeometry",
    "rtkdrawgeometricphantom",
    "rtkdrawshepploganphantom",
    "rtkelektasynergygeometry",
    "rtkfdk",
    "rtkforwardprojections",
    "rtkorageometry",
    "rtkprojectgeometricphantom",
    "rtkprojections",
    "rtkprojectshepploganphantom",
    "rtkshowgeometry",
    "rtksimulatedgeometry",
    "rtkvarianobigeometry",
]

# Dynamically access make_application_func from rtkExtras
rtk_extras = importlib.import_module("itk.rtkExtras")
make_application_func = getattr(rtk_extras, "make_application_func")

# Dynamically register applications
for app_name in _app_modules:
    setattr(rtk_module, app_name, make_application_func(app_name))
