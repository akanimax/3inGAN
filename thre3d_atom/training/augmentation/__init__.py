"""
    ==================================================================================================
    Package contains modules for applying differentiable augmentation.
    Code has been taken from Nvidia's official
    stylegan2-ada-pytorch repository (https://github.com/NVlabs/stylegan2-ada-pytorch)

    Our changes:
        1. Minor code formatting
        2. Stripping off the custom CUDA code, so performance of the package here would be lower
           compared to the original one.
    ==================================================================================================
"""
