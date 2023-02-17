__all__ = [
    'add_rtkprojectors_group',
    'SetForwardProjectionFromArgParse',
    'SetBackProjectionFromArgParse',
]

# Mimicks rtkprojectors_section.ggo
def add_rtkprojectors_group(parser):
  rtkprojectors_group = parser.add_argument_group('Projectors')
  rtkprojectors_group.add_argument('--fp', '-f', help='Forward projection method',
                          choices=['Joseph','CudaRayCast','JosephAttenuated','Zeng'],
                          default='Joseph')
  rtkprojectors_group.add_argument('--bp', '-b', help='Back projection method',
                          choices=['VoxelBasedBackProjection','Joseph','CudaVoxelBased','CudaRayCast','JosephAttenuated', 'Zeng'],
                          default='VoxelBasedBackProjection')
  rtkprojectors_group.add_argument('--attenuationmap', help='Attenuation map relative to the volume to perfom the attenuation correction (JosephAttenuated and Zeng)')
  rtkprojectors_group.add_argument('--sigmazero', help='PSF value at a distance of 0 meter of the detector (Zeng only)', type=float)
  rtkprojectors_group.add_argument('--alphapsf', help='Slope of the PSF against the detector distance (Zeng only)', type=float)
  rtkprojectors_group.add_argument('--inferiorclipimage', help='Inferior clip of the ray for each pixel of the projections (Joseph only)')
  rtkprojectors_group.add_argument('--superiorclipimage', help='Superior clip of the ray for each pixel of the projections (Joseph only)')

# Mimicks SetBackProjectionFromGgo
def SetBackProjectionFromArgParse(args_info, recon):
  if args_info.attenuationmap is not None:
    attenuation_map = itk.imread(args_info.attenuationmap)
  if args_info.inferiorclipimage is not None:
    inferior_clip_image = itk.imread(args_info.inferiorclipimage)
  if args_info.superiorclipimage is not None:
    superior_clip_image = itk.imread(args_info.superiorclipimage)
  ReconType = type(recon)
  if args_info.bp == 'VoxelBasedBackProjection': # bp_arg_VoxelBasedBackProjection
    recon.SetBackProjectionFilter(ReconType.BackProjectionType_BP_VOXELBASED)
  elif args_info.bp == 'Joseph': # bp_arg_Joseph
    recon.SetBackProjectionFilter(ReconType.BackProjectionType_BP_JOSEPH)
  elif args_info.bp == 'CudaVoxelBased': # bp_arg_CudaVoxelBased
    recon.SetBackProjectionFilter(ReconType.BackProjectionType_BP_CUDAVOXELBASED)
  elif args_info.bp == 'CudaRayCast': # bp_arg_CudaRayCast
    recon.SetBackProjectionFilter(ReconType.BackProjectionType_BP_CUDARAYCAST)
  elif args_info.bp == 'JosephAttenuated': # bp_arg_JosephAttenuated
    recon.SetBackProjectionFilter(ReconType.BackProjectionType_BP_JOSEPHATTENUATED)
    if args_info.inferiorclipimage is not None:
      recon.SetInferiorClipImage(inferior_clip_image)
    if args_info.superiorclipimage is not None:
      recon.SetSuperiorClipImage(superior_clip_image)
    if args_info.attenuationmap is not None:
      recon.SetAttenuationMap(attenuation_map)
  elif args_info.bp == 'Zeng': # bp_arg_RotationBased
    recon.SetBackProjectionFilter(ReconType.BackProjectionType_BP_ZENG)
    if args_info.sigmazero is not None:
      recon.SetSigmaZero(args_info.sigmazero);
    if args_info.alphapsf is not None:
      recon.SetAlphaPSF(args_info.alphapsf);
    if args_info.attenuationmap is not None:
      recon.SetAttenuationMap(attenuationMap);

# Mimicks SetForwardProjectionFromGgo
def SetForwardProjectionFromArgParse(args_info, recon):
  if args_info.attenuationmap is not None:
    attenuation_map = itk.imread(args_info.attenuationmap)
  ReconType = type(recon)
  if args_info.fp == 'Joseph': # fp_arg_Joseph
    recon.SetForwardProjectionFilter(ReconType.ForwardProjectionType_FP_JOSEPH)
  elif args_info.fp == 'CudaRayCast': # fp_arg_CudaRayCast
    recon.SetForwardProjectionFilter(ReconType.ForwardProjectionType_FP_CUDARAYCAST)
  elif args_info.fp == 'JosephAttenuated': # fp_arg_JosephAttenuated
    recon.SetForwardProjectionFilter(ReconType.ForwardProjectionType_FP_JOSEPHATTENUATED)
    if args_info.attenuationmap is not None:
      recon.SetAttenuationMap(attenuation_map)
  elif args_info.fp == 'Zeng': # fp_arg_RotationBased
    recon.SetForwardProjectionFilter(ReconType.ForwardProjectionType_FP_ZENG)
    if args_info.sigmazero is not None:
      recon.SetSigmaZero(args_info.sigmazero);
    if args_info.alphapsf is not None:
      recon.SetAlphaPSF(args_info.alphapsf);
    if args_info.attenuationmap is not None:
      recon.SetAttenuationMap(attenuationMap);
