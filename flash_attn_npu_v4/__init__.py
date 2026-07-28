__version__ = "0.1.1"

import torch_npu


def is_arch22() -> bool:
    """Return True if the current device belongs to arch22."""
    device_name = torch_npu.npu.get_device_name()
    return "Ascend910B" in device_name or "Ascend910C" in device_name


def is_arch35() -> bool:
    """Return True if the current device belongs to arch35."""
    device_name = torch_npu.npu.get_device_name()
    return "Ascend950" in device_name


if is_arch22():
    from .interface_arch22 import flash_attn_varlen_func
elif is_arch35():
    from .interface_arch35 import flash_attn_varlen_func
else:
    raise RuntimeError(f"Unsupported Ascend device: {torch_npu.npu.get_device_name()}")
