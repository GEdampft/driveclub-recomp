"""memscan.py: live memory of the running game (bb-probe.exe), for the VR work.

  memscan.py find F1 F2 ... [--tol T]     addresses where these little-endian floats sit consecutively (relative T, 1e-4)
  memscan.py read ADDR COUNT              COUNT floats from ADDR (hex), 4 per line
  memscan.py region ADDR                  the memory region holding ADDR (base, size, protection)
"""
import ctypes
import ctypes.wintypes as wt
import struct
import subprocess
import sys

import numpy as np

k32 = ctypes.WinDLL('kernel32', use_last_error=True)
PROCESS_VM_READ, PROCESS_QUERY_INFORMATION = 0x10, 0x400


class MBI(ctypes.Structure):
    _fields_ = [('BaseAddress', ctypes.c_uint64), ('AllocationBase', ctypes.c_uint64), ('AllocationProtect', wt.DWORD),
                ('PartitionId', wt.WORD), ('RegionSize', ctypes.c_uint64), ('State', wt.DWORD), ('Protect', wt.DWORD),
                ('Type', wt.DWORD)]


def pid():
    out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq bb-probe.exe', '/FO', 'CSV', '/NH'], capture_output=True,
                         text=True).stdout
    return int(out.split(',')[1].strip('"'))


H = k32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid())


def read(addr, size):
    buf = ctypes.create_string_buffer(size)
    n = ctypes.c_size_t()
    if not k32.ReadProcessMemory(H, ctypes.c_void_p(addr), buf, size, ctypes.byref(n)):
        return b''
    return buf.raw[:n.value]


def regions():
    addr, mbi = 0, MBI()
    while k32.VirtualQueryEx(H, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
        if mbi.State == 0x1000 and mbi.Protect & 0xEE and not mbi.Protect & 0x500:  # committed, readable; no guard, no write-combined (GPU upload memory)
            yield mbi.BaseAddress, mbi.RegionSize, mbi.Protect, mbi.AllocationBase
        addr = mbi.BaseAddress + mbi.RegionSize
        if addr >= 1 << 47:
            break


def find(values, tol):
    want = np.array(values, dtype=np.float32)
    hits = 0
    for base, size, prot, alloc in regions():
        for off in range(0, size, 1 << 26):
            data = read(base + off, min(1 << 26, size - off))
            if len(data) < 4 * len(want):
                continue
            a = np.frombuffer(data[:len(data) // 4 * 4], dtype='<f4')
            with np.errstate(all='ignore'):
                m = np.abs(a[:len(a) - len(want) + 1] - want[0]) <= np.abs(want[0]) * tol + 1e-7
                idx = np.nonzero(m)[0]
                for i in idx:
                    seg = a[i:i + len(want)]
                    if np.all(np.abs(seg - want) <= np.abs(want) * tol + 1e-7):
                        addr = base + off + int(i) * 4
                        print(f'{addr:#x}  region {base:#x}+{size:#x} prot {prot:#x} alloc {alloc:#x}')
                        hits += 1
                        if hits > 200:
                            return


def main():
    cmd = sys.argv[1]
    if cmd == 'find':
        args = sys.argv[2:]
        tol = 1e-4
        if '--tol' in args:
            i = args.index('--tol')
            tol = float(args[i + 1])
            args = args[:i] + args[i + 2:]
        find([float(v) for v in args], tol)
    elif cmd == 'read':
        addr, count = int(sys.argv[2], 16), int(sys.argv[3])
        data = read(addr, count * 4)
        f = struct.unpack(f'<{len(data) // 4}f', data)
        for i in range(0, len(f), 4):
            print(f'{addr + i * 4:#x}: ' + ' '.join(f'{x:12.6g}' for x in f[i:i + 4]))
    elif cmd == 'region':
        mbi = MBI()
        k32.VirtualQueryEx(H, ctypes.c_void_p(int(sys.argv[2], 16)), ctypes.byref(mbi), ctypes.sizeof(mbi))
        print(f'base {mbi.BaseAddress:#x} size {mbi.RegionSize:#x} alloc {mbi.AllocationBase:#x} prot {mbi.Protect:#x}')


if __name__ == '__main__':
    main()
