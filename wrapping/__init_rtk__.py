import sys
import importlib
import os
import glob

itk_module = sys.modules["itk"]
rtk_module = getattr(itk_module, "RTK")

# Import RTK submodules
rtk_submodules = [
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

# Dynamically build application modules list from applications/rtk*/rtk*.py
_app_modules = []
base_dir = os.path.dirname(__file__)
apps_root = os.path.join(base_dir, "applications")
pattern = os.path.join(apps_root, "rtk*", "rtk*.py")
for filepath in glob.glob(pattern):
    name = os.path.splitext(os.path.basename(filepath))[0]
    _app_modules.append(name)
_app_modules.sort()

# Dynamically access make_application_func from rtkExtras
rtk_extras = importlib.import_module("itk.rtkExtras")
make_application_func = getattr(rtk_extras, "make_application_func")

# Dynamically register applications
for app_name in _app_modules:
    setattr(rtk_module, app_name, make_application_func(app_name))
