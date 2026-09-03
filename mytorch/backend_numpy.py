"""Backward-compatible imports for the original NumPy backend module."""

from .backend import Device, all_devices, cpu, default_device

CPUDevice = Device
