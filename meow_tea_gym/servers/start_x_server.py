#!/usr/bin/env python
# copied from https://github.com/askforalfred/alfred/blob/master/scripts/startx.py

import subprocess
import shlex
import re
import platform
import tempfile
import os
import sys
import argparse

def pci_records():
    records = []
    command = shlex.split('lspci -vmm')
    output = subprocess.check_output(command).decode()

    for devices in output.strip().split("\n\n"):
        record = {}
        records.append(record)
        for row in devices.split("\n"):
            key, value = row.split("\t")
            record[key.split(':')[0]] = value

    return records

def generate_xorg_conf(devices, gpu_id=None):
    """
    Generate Xorg configuration.

    Args:
        devices: List of all GPU bus IDs
        gpu_id: If specified, only configure this specific GPU (0-indexed).
                If None, configure all GPUs (old behavior).
    """
    xorg_conf = []

    device_section = """
Section "Device"
    Identifier     "Device{device_id}"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "{bus_id}"
EndSection
"""
    server_layout_section = """
Section "ServerLayout"
    Identifier     "Layout0"
    {screen_records}
EndSection
"""
    screen_section = """
Section "Screen"
    Identifier     "Screen{screen_id}"
    Device         "Device{device_id}"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    SubSection     "Display"
        Depth       24
        Virtual 1024 768
    EndSubSection
EndSection
"""

    # If gpu_id is specified, only use that specific GPU
    if gpu_id is not None:
        if gpu_id >= len(devices):
            raise ValueError(f"GPU ID {gpu_id} is out of range. Available GPUs: 0-{len(devices)-1}")
        devices_to_use = [devices[gpu_id]]
        device_ids = [0]  # Always use device 0 for single GPU config
    else:
        devices_to_use = devices
        device_ids = range(len(devices))

    screen_records = []
    for i, (device_id, bus_id) in enumerate(zip(device_ids, devices_to_use)):
        xorg_conf.append(device_section.format(device_id=i, bus_id=bus_id))
        xorg_conf.append(screen_section.format(device_id=i, screen_id=i))
        screen_records.append('Screen {screen_id} "Screen{screen_id}" 0 0'.format(screen_id=i))

    xorg_conf.append(server_layout_section.format(screen_records="\n    ".join(screen_records)))

    output =  "\n".join(xorg_conf)
    print(output)
    return output

def startx(display, gpu_id=None):
    """
    Start X server on specified display.

    Args:
        display: X display number
        gpu_id: GPU ID to use (0-indexed). If None, use all GPUs.
    """
    if platform.system() != 'Linux':
        raise Exception("Can only run startx on linux")

    devices = []
    for r in pci_records():
        if r.get('Vendor', '') == 'NVIDIA Corporation' \
                and r['Class'] in ['VGA compatible controller', '3D controller']:
            bus_id = 'PCI:' + ':'.join(map(lambda x: str(int(x, 16)), re.split(r'[:\.]', r['Slot'])))
            devices.append(bus_id)

    if not devices:
        raise Exception("no nvidia cards found")

    print(f"Found {len(devices)} NVIDIA GPU(s)")
    if gpu_id is not None:
        print(f"Configuring X server to use GPU {gpu_id}")

    try:
        fd, path = tempfile.mkstemp()
        with open(path, "w") as f:
            f.write(generate_xorg_conf(devices, gpu_id=gpu_id))
        command = shlex.split("Xorg -noreset +extension GLX +extension RANDR +extension RENDER -config %s :%s" % (path, display))
        subprocess.call(command)
    finally:
        os.close(fd)
        os.unlink(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start X server on specified display')
    parser.add_argument('--display', '-d', type=int, default=7,
                        help='Display number to use (default: 7)')
    parser.add_argument('--gpu', '-g', type=int, default=None,
                        help='GPU ID to use (0-indexed). If not specified, uses all GPUs.')
    args = parser.parse_args()

    print("Starting X on DISPLAY=:%s" % args.display)
    if args.gpu is not None:
        print("Using GPU %s" % args.gpu)
    startx(args.display, gpu_id=args.gpu)