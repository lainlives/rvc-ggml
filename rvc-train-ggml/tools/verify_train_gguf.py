#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_train_gguf.py -- standalone training-state GGUF verification (tool)

Two purposes (can be used together):
  1) --pth with --gguf: re-parse the GGUF (hand-written reader, standard library only), and for each tensor
     do a **byte-level** comparison against the .pth state_dict (converted per rvc.train.out_dtype); also check the tensor set and metadata.
  2) --pth with --hashes (output of C++ `rvc-train --load-gguf --dump-hashes`):
     recompute CRC32 and ne/nbytes to verify that the C++ load path matches tensor by tensor.

Usage:
  python tools\\verify_train_gguf.py --pth <in.pth> --gguf <out.gguf> [--hashes <dump.txt>]
Exit codes: 0 = all passed; 1 = mismatch found; 2 = argument/file error.
"""

import argparse
import os
import struct
import sys
import zlib
from collections import OrderedDict

import torch

ALIGN_DEFAULT = 32


# ---------------------------------------------------------------- GGUF reading

class GgufReader:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = f.read()
        self.path = path
        if self.data[0:4] != b'GGUF':
            raise RuntimeError('not a GGUF file: %s' % path)
        off = 4
        self.version, off = self._u32(off)
        self.tensor_count, off = self._u64(off)
        kv_count, off = self._u64(off)
        if self.version not in (2, 3):
            raise RuntimeError('unsupported GGUF version: %d' % self.version)

        self.kv = OrderedDict()
        for _ in range(kv_count):
            key, off = self._str(off)
            type_id, off = self._i32(off)
            value, off = self._value(off, type_id)
            self.kv[key] = (type_id, value)

        self.tensors = OrderedDict()  # name -> dict(ne, type, offset)
        for _ in range(self.tensor_count):
            name, off = self._str(off)
            n_dims, off = self._u32(off)
            ne = []
            for _ in range(n_dims):
                d, off = self._i64(off)
                ne.append(d)
            type_id, off = self._i32(off)
            t_off, off = self._u64(off)
            self.tensors[name] = {'ne': ne, 'type': type_id, 'offset': t_off}

        align = self.kv.get('general.alignment', (None, ALIGN_DEFAULT))[1]
        self.alignment = int(align)
        self.data_start = (off + self.alignment - 1) // self.alignment * self.alignment

    # -- Basic reads --
    def _u32(self, off):
        return struct.unpack_from('<I', self.data, off)[0], off + 4

    def _i32(self, off):
        return struct.unpack_from('<i', self.data, off)[0], off + 4

    def _u64(self, off):
        return struct.unpack_from('<Q', self.data, off)[0], off + 8

    def _i64(self, off):
        return struct.unpack_from('<q', self.data, off)[0], off + 8

    def _str(self, off):
        n, off = self._u64(off)
        s = self.data[off:off + n].decode('utf-8')
        return s, off + n

    def _value(self, off, type_id):
        if type_id == 0:   # U8
            return self.data[off], off + 1
        if type_id == 1:   # I8
            return struct.unpack_from('<b', self.data, off)[0], off + 1
        if type_id == 2:   # U16
            return struct.unpack_from('<H', self.data, off)[0], off + 2
        if type_id == 3:   # I16
            return struct.unpack_from('<h', self.data, off)[0], off + 2
        if type_id == 4:   # U32
            return self._u32(off)
        if type_id == 5:   # I32
            return self._i32(off)
        if type_id == 6:   # F32
            return struct.unpack_from('<f', self.data, off)[0], off + 4
        if type_id == 7:   # BOOL
            return self.data[off] != 0, off + 1
        if type_id == 8:   # STRING
            return self._str(off)
        if type_id == 9:   # ARRAY
            elem_type, off = self._i32(off)
            count, off = self._u64(off)
            items = []
            for _ in range(count):
                v, off = self._value(off, elem_type)
                items.append(v)
            return items, off
        if type_id == 10:  # U64
            return self._u64(off)
        if type_id == 11:  # I64
            return self._i64(off)
        if type_id == 12:  # F64
            return struct.unpack_from('<d', self.data, off)[0], off + 8
        raise RuntimeError('unknown GGUF metadata type: %d' % type_id)

    def tensor_bytes(self, name):
        t = self.tensors[name]
        nbytes = 1
        for d in t['ne']:
            nbytes *= int(d)
        if t['type'] == 0:
            nbytes *= 4
        elif t['type'] == 1:
            nbytes *= 2
        else:
            raise RuntimeError('unsupported tensor type: %d' % t['type'])
        start = self.data_start + t['offset']
        return self.data[start:start + nbytes]


# ---------------------------------------------------------------- pth reading

def load_state_dict(path):
    d = torch.load(path, map_location='cpu')
    if not isinstance(d, dict):
        raise RuntimeError('top level is not a dict')
    if 'model' in d and isinstance(d['model'], dict):
        return d, d['model'], 'train'
    if 'weight' in d and isinstance(d['weight'], dict):
        return d, d['weight'], 'small'
    if d and all(torch.is_tensor(v) for v in d.values()):
        return d, d, 'raw'
    raise RuntimeError('unrecognized state_dict')


def to_ne(shape):
    if len(shape) == 0:
        return [1]
    return [int(x) for x in reversed(shape)]


def convert_tensor(t, type_id):
    a = t.detach().cpu().contiguous()
    if type_id == 1:   # F16
        return a.half().numpy().tobytes(order='C')
    return a.float().numpy().tobytes(order='C')


# ---------------------------------------------------------------- Main flow

def main():
    ap = argparse.ArgumentParser(description='training-state GGUF verification (pth byte-level comparison + C++ CRC comparison)')
    ap.add_argument('--pth', required=True)
    ap.add_argument('--gguf', default='')
    ap.add_argument('--hashes', default='')
    args = ap.parse_args()

    if not args.gguf and not args.hashes:
        print('[error] provide at least --gguf or --hashes', file=sys.stderr)
        return 2

    top, sd, form = load_state_dict(args.pth)
    print('[verify] pth=%s form=%s tensors=%d' % (os.path.abspath(args.pth), form, len(sd)))

    ok = True

    # ---- 1) GGUF byte-level comparison ----
    if args.gguf:
        g = GgufReader(args.gguf)
        print('[verify] gguf=%s version=%d tensors=%d' % (os.path.abspath(args.gguf), g.version, len(g.tensors)))

        out_dtype = g.kv.get('rvc.train.out_dtype', (None, 'F32'))[1]
        print('[verify] metadata kind=%s version=%s has_f0=%s dtype=%s sr=%s spec_channels=%s hop=%s' % (
            g.kv.get('rvc.train.kind', (None, '?'))[1],
            g.kv.get('rvc.train.version', (None, '?'))[1],
            g.kv.get('rvc.train.has_f0', (None, '?'))[1],
            out_dtype,
            g.kv.get('rvc.train.sample_rate', (None, '?'))[1],
            g.kv.get('rvc.train.spec_channels', (None, '?'))[1],
            g.kv.get('rvc.train.hop_length', (None, '?'))[1]))

        pth_names = set(sd.keys())
        gguf_names = set(g.tensors.keys())
        missing = sorted(pth_names - gguf_names)
        extra = sorted(gguf_names - pth_names)
        if missing:
            ok = False
            print('[verify] missing tensors: %d (first 5): %s' % (len(missing), missing[:5]))
        if extra:
            ok = False
            print('[verify] extra tensors: %d (first 5): %s' % (len(extra), extra[:5]))
        if not missing and not extra:
            print('[verify] tensor sets match: %d' % len(pth_names))

        mismatch = []
        shape_bad = []
        for name, t in sd.items():
            if name not in g.tensors:
                continue
            info = g.tensors[name]
            ne_got = info['ne']
            ne_want = to_ne(t.shape)
            if ne_got != ne_want:
                shape_bad.append((name, ne_want, ne_got))
                continue
            want = convert_tensor(t, info['type'])
            got = g.tensor_bytes(name)
            if want != got:
                mismatch.append(name)

        if shape_bad:
            ok = False
            print('[verify] shape mismatches: %d (first 5): %s' % (len(shape_bad), shape_bad[:5]))
        if mismatch:
            ok = False
            print('[verify] byte mismatches: %d (first 5): %s' % (len(mismatch), mismatch[:5]))
        if not shape_bad and not mismatch:
            print('[verify] per-tensor byte-level comparison: all match (%d/%d, output %s)' % (len(sd), len(sd), out_dtype))

    # ---- 2) C++ CRC comparison ----
    if args.hashes:
        # Output tensor type: inferred from GGUF metadata/first line
        out_type = 1 if (args.gguf and GgufReader(args.gguf).kv.get('rvc.train.out_dtype', (None, 'F32'))[1] == 'F16') else 0

        cpp = OrderedDict()
        with open(args.hashes, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if len(parts) != 7:
                    continue
                name = parts[0]
                ne = [int(x) for x in parts[1:5]]
                nbytes = int(parts[5])
                crc = int(parts[6], 16)
                cpp[name] = (ne, nbytes, crc)
        if not cpp:
            print('[verify] hashes file is empty or malformed: %s' % args.hashes)
            ok = False
        else:
            bad = []
            for name, (ne, nbytes, crc) in cpp.items():
                if name not in sd:
                    bad.append((name, 'not in pth'))
                    continue
                t = sd[name]
                ne_want = to_ne(t.shape) + [1] * (4 - len(to_ne(t.shape)))
                if ne_want != ne:
                    bad.append((name, 'ne %s != %s' % (ne, ne_want)))
                    continue
                data = convert_tensor(t, out_type)
                if len(data) != nbytes:
                    bad.append((name, 'nbytes %d != %d' % (nbytes, len(data))))
                    continue
                crc_want = zlib.crc32(data) & 0xFFFFFFFF
                if crc_want != crc:
                    bad.append((name, 'crc32 %08X != %08X' % (crc, crc_want)))
            if bad:
                ok = False
                print('[verify] C++ CRC mismatches: %d (first 5): %s' % (len(bad), bad[:5]))
            else:
                print('[verify] C++ CRC comparison: all match (%d/%d)' % (len(cpp), len(cpp)))

    print('[verify] %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print('[error] %s' % e, file=sys.stderr)
        sys.exit(2)
