import pytest
import itk
from itk import RTK as rtk


@pytest.mark.parametrize("direction", ["Forward", "Back"])
def test_virtual_detector_padding_api(direction):
    image_type = itk.Image[itk.F, 3]
    projector_type = getattr(rtk, f"Zeng{direction}ProjectionImageFilter")
    projector = projector_type[image_type, image_type].New()
    assert projector.GetVirtualDetectorPadding() == 0
    for padding in [1, 22, 120, 0]:
        projector.SetVirtualDetectorPadding(padding)
        assert projector.GetVirtualDetectorPadding() == padding


@pytest.mark.parametrize("direction", ["Forward", "Back"])
def test_cuda_virtual_detector_padding_api(direction):
    projector_type = getattr(rtk, f"CudaZeng{direction}ProjectionImageFilter", None)
    if projector_type is None:
        pytest.skip("RTK was built without CUDA")
    projector = projector_type.New()
    assert projector.GetVirtualDetectorPadding() == 0
    projector.SetVirtualDetectorPadding(22)
    assert projector.GetVirtualDetectorPadding() == 22
