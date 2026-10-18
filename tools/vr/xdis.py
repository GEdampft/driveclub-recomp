"""xdis.py ADDR [BYTES_BEFORE] [BYTES_AFTER]: x86-64 disassembly of the running game (bb-probe.exe) around ADDR."""
import sys
import capstone
import memscan
a = int(sys.argv[1], 16); before = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0x80; after = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0x60
data = memscan.read(a - before, before + after)
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.skipdata = True
for ins in md.disasm(data, a - before):
    mark = '>>' if ins.address == a else '  '
    print(f'{mark}{ins.address:#x} (+{ins.address - 0x800000000:#x}): {ins.mnemonic} {ins.op_str}')
