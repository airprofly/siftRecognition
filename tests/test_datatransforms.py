
import numpy as np
import torch

# import torch.nn as nn
from PIL import Image

from data_transforms import get_fundamental_transforms

def test_fundamental_transforms():
    """
    Tests the transforms using output from disk
    """

    transforms = get_fundamental_transforms(
        inp_size=(100, 50), pixel_mean=np.array([0.5]), pixel_std=np.array([0.3])
    )

    try:
        inp_img = Image.fromarray(
            np.loadtxt("proj4_unit_tests/data/transform_inp.txt", dtype="uint8")
        )
        output_img = transforms(inp_img)
        expected_output = torch.load("proj4_unit_tests/data/transform_out.pt")

    except:
        inp_img = Image.fromarray(
            np.loadtxt("../proj4_unit_tests/data/transform_inp.txt", dtype="uint8")
        )
        output_img = transforms(inp_img)
        expected_output = torch.load("../proj4_unit_tests/data/transform_out.pt")

    assert isinstance(output_img, torch.Tensor)
    assert torch.allclose(expected_output, output_img)
