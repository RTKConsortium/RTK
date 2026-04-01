import sys
import importlib

itk_module = sys.modules["itk"]
rtk_module = getattr(itk_module, "RTK")

# Load the CMake-generated version and assign it to `itk.RTK.__version__`.
rtk_version = importlib.import_module("itk.rtkConfig").RTK_GLOBAL_VERSION_STRING
setattr(rtk_module, "__version__", rtk_version)

# Import RTK submodules
rtk_submodules = [
    "itk.rtkargumentparser",
    "itk.rtkinputprojections_group",
    "itk.rtk3Doutputimage_group",
    "itk.rtk4Doutputimage_group",
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
    "rtkadmmwavelets",
    "rtkamsterdamshroud",
    "rtkbackprojections",
    "rtkbioscangeometry",
    "rtkcheckimagequality",
    "rtkconjugategradient",
    "rtkdigisensgeometry",
    "rtkdrawgeometricphantom",
    "rtkdrawshepploganphantom",
    "rtkelektasynergygeometry",
    "rtkextractphasesignal",
    "rtkfdk",
    "rtkfieldofview",
    "rtkforwardprojections",
    "rtkfourdconjugategradient",
    "rtkfourdfdk",
    "rtkfourdrooster",
    "rtkfourdsart",
    "rtki0estimation",
    "rtkimagxgeometry",
    "rtkiterativefdk",
    "rtklagcorrection",
    "rtkmaskcollimation",
    "rtkgaincorrection",
    "rtkorageometry",
    "rtkosem",
    "rtkprojectgeometricphantom",
    "rtkprojections",
    "rtkprojectshepploganphantom",
    "rtkregularizedconjugategradient",
    "rtkscatterglarecorrection",
    "rtkshowgeometry",
    "rtksart",
    "rtksimulatedgeometry",
    "rtksubselect",
    "rtktotalnuclearvariationdenoising",
    "rtktotalvariationdenoising",
    "rtkvarianobigeometry",
    "rtkwaveletsdenoising",
    "rtkxradgeometry",
]

# Dynamically access make_application_func from rtkExtras
rtk_extras = importlib.import_module("itk.rtkExtras")
make_application_func = getattr(rtk_extras, "make_application_func")

# Dynamically register applications
for app_name in _app_modules:
    setattr(rtk_module, app_name, make_application_func(app_name))
