import sys
import importlib

itk_module = sys.modules['itk']
rtk_module = getattr(itk_module, 'RTK')
rtk_submodules = ['itk.rtkinputprojections_group',
                  'itk.rtk3Doutputimage_group',
                  'itk.rtkprojectors_group',
                  'itk.rtkiterations_group']
for mod_name in rtk_submodules:
  mod = importlib.import_module(mod_name)
  for a in dir(mod):
    if a[0] != '_':
      setattr(rtk_module, a, getattr(mod, a))
