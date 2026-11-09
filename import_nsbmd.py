import math
import struct

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix

from .util import safe_register_class
from .nns_material import compute_mat_type, generate_nodes


class NsbmdError(Exception):
    pass


def _u16(d, o):
    return struct.unpack_from('<H', d, o)[0]


def _s16(d, o):
    return struct.unpack_from('<h', d, o)[0]


def _u32(d, o):
    return struct.unpack_from('<I', d, o)[0]


def _s32(d, o):
    return struct.unpack_from('<i', d, o)[0]


def _signed(raw, bits):
    raw &= (1 << bits) - 1
    if raw & (1 << (bits - 1)):
        raw -= 1 << bits
    return raw


def decompress_lz10(data):
    size = data[1] | (data[2] << 8) | (data[3] << 16)
    out = bytearray()
    p = 4
    while len(out) < size:
        flags = data[p]
        p += 1
        for bit in range(8):
            if len(out) >= size:
                break
            if flags & (0x80 >> bit):
                b1 = data[p]
                b2 = data[p + 1]
                p += 2
                length = (b1 >> 4) + 3
                disp = (((b1 & 0x0F) << 8) | b2) + 1
                if disp > len(out):
                    raise NsbmdError("The compressed data is corrupted.")
                for _ in range(length):
                    out.append(out[-disp])
            else:
                out.append(data[p])
                p += 1
    return bytes(out[:size])


def load_file(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    if data[:4] == b'LZ77':
        data = data[4:]
    if data[:4] != b'BMD0' and data[:1] == b'\x10':
        try:
            data = decompress_lz10(data)
        except IndexError:
            raise NsbmdError("The compressed data is truncated.")
    if data[:4] != b'BMD0':
        raise NsbmdError("This isn't an NSBMD file (no BMD0 signature, "
                         "compressed or not).")
    return data


def read_dict(d, off):
    count = d[off + 1]
    entries_off = off + _u16(d, off + 6)
    entry_size = _u16(d, entries_off)
    names_off = entries_off + _u16(d, entries_off + 2)
    entries = [entries_off + 4 + i * entry_size for i in range(count)]
    names = []
    for i in range(count):
        raw = d[names_off + i * 16:names_off + i * 16 + 16].split(b'\x00')[0]
        names.append(raw.decode('shift_jis', 'replace'))
    return names, entries


def _identity():
    return [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]


def _mul(a, b):
    return [[a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c]
             + a[r][3] * b[3][c] for c in range(4)] for r in range(4)]


def _apply(m, v):
    return tuple(m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] + m[r][3]
                 for r in range(3))


def _normal_matrix(m):
    a = [[m[r][c] for c in range(3)] for r in range(3)]
    cof = [[a[(r + 1) % 3][(c + 1) % 3] * a[(r + 2) % 3][(c + 2) % 3]
            - a[(r + 1) % 3][(c + 2) % 3] * a[(r + 2) % 3][(c + 1) % 3]
            for c in range(3)] for r in range(3)]
    det = a[0][0] * cof[0][0] + a[0][1] * cof[0][1] + a[0][2] * cof[0][2]
    sign = -1.0 if det < 0 else 1.0
    return [[cof[r][c] * sign for c in range(3)] for r in range(3)]


def _apply_normal(n, v):
    x, y, z = (n[r][0] * v[0] + n[r][1] * v[1] + n[r][2] * v[2]
               for r in range(3))
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-12:
        return (0.0, 0.0, 1.0)
    return (x / length, y / length, z / length)


def _affine_inverse(m):
    a = [[m[r][c] for c in range(3)] for r in range(3)]
    cof = [[a[(r + 1) % 3][(c + 1) % 3] * a[(r + 2) % 3][(c + 2) % 3]
            - a[(r + 1) % 3][(c + 2) % 3] * a[(r + 2) % 3][(c + 1) % 3]
            for c in range(3)] for r in range(3)]
    det = a[0][0] * cof[0][0] + a[0][1] * cof[0][1] + a[0][2] * cof[0][2]
    if abs(det) < 1e-12:
        return _identity()
    inv = [[cof[c][r] / det for c in range(3)] for r in range(3)]
    t = [m[r][3] for r in range(3)]
    out = _identity()
    for r in range(3):
        for c in range(3):
            out[r][c] = inv[r][c]
        out[r][3] = -(inv[r][0] * t[0] + inv[r][1] * t[1] + inv[r][2] * t[2])
    return out


def pivot_matrix(index, neg, a, b):
    one = -1.0 if neg & 1 else 1.0
    c = -b if neg & 2 else b
    dd = -a if neg & 4 else a
    m = [[0.0] * 3 for _ in range(3)]
    row, col = index // 3, index % 3
    m[row][col] = one
    rows = [r for r in range(3) if r != row]
    cols = [k for k in range(3) if k != col]
    m[rows[0]][cols[0]] = a
    m[rows[0]][cols[1]] = b
    m[rows[1]][cols[0]] = c
    m[rows[1]][cols[1]] = dd
    return m


def read_node(d, off):
    flags = _u16(d, off)
    m00 = _s16(d, off + 2) / 4096.0
    p = off + 4
    translate = (0.0, 0.0, 0.0)
    if not flags & 0x0001:
        translate = tuple(_s32(d, p + 4 * k) / 4096.0 for k in range(3))
        p += 12
    rotate = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if not flags & 0x0002:
        if flags & 0x0008:
            a = _s16(d, p) / 4096.0
            b = _s16(d, p + 2) / 4096.0
            p += 4
            rotate = pivot_matrix((flags >> 4) & 0x0F, (flags >> 8) & 0x07,
                                  a, b)
        else:
            values = [m00] + [_s16(d, p + 2 * k) / 4096.0 for k in range(8)]
            p += 16
            rotate = [values[0:3], values[3:6], values[6:9]]
    scale = (1.0, 1.0, 1.0)
    if not flags & 0x0004:
        scale = tuple(_s32(d, p + 4 * k) / 4096.0 for k in range(3))
        p += 24
    return {'flags': flags, 'translate': translate, 'rotate': rotate,
            'scale': scale}


def node_local_matrix(node, parent_scale=None):
    r = node['rotate']
    t = node['translate']
    s = node['scale']
    m = [[r[c][row] * s[c] for c in range(3)] + [t[row]] for row in range(3)]
    m.append([0.0, 0.0, 0.0, 1.0])
    if parent_scale is not None:
        for row in range(3):
            for c in range(3):
                m[row][c] /= parent_scale[row] if parent_scale[row] else 1.0
    return m


_TILING = ('clamp', 'repeat', 'clamp', 'flip')
_POLYGON_MODES = ('modulate', 'decal', 'toon_highlight', 'shadow')
_TEX_GEN_MODES = ('none', 'tex', 'nrm', 'pos')


def _color5(v):
    return ((v & 0x1F), ((v >> 5) & 0x1F), ((v >> 10) & 0x1F))


def read_material(d, name, off):
    size = _u16(d, off + 2)
    dif_amb = _u32(d, off + 4)
    spe_emi = _u32(d, off + 8)
    poly_attr = _u32(d, off + 12)
    tex_param = _u32(d, off + 20)
    flags = _u16(d, off + 30)
    material = {
        'name': name,
        'diffuse': _color5(dif_amb),
        'ambient': _color5(dif_amb >> 16),
        'specular': _color5(spe_emi),
        'emission': _color5(spe_emi >> 16),
        'use_srst': bool(spe_emi & 0x8000),
        'lights': [bool(poly_attr & (1 << k)) for k in range(4)],
        'polygon_mode': _POLYGON_MODES[(poly_attr >> 4) & 3],
        'back': bool(poly_attr & 0x40),
        'front': bool(poly_attr & 0x80),
        'update_depth': bool(poly_attr & 0x800),
        'far_clip': bool(poly_attr & 0x1000),
        'one_dot': bool(poly_attr & 0x2000),
        'depth_equal': bool(poly_attr & 0x4000),
        'fog': bool(poly_attr & 0x8000),
        'alpha': (poly_attr >> 16) & 0x1F,
        'polygon_id': (poly_attr >> 24) & 0x3F,
        'tiling_u': _TILING[((tex_param >> 16) & 1) | (((tex_param >> 18) & 1) << 1)],
        'tiling_v': _TILING[((tex_param >> 17) & 1) | (((tex_param >> 19) & 1) << 1)],
        'tex_gen_mode': _TEX_GEN_MODES[(tex_param >> 30) & 3],
        'wireframe': bool(flags & 0x0020),
        'tex_width': _u16(d, off + 32),
        'tex_height': _u16(d, off + 34),
        'tex_scale': (1.0, 1.0),
        'tex_rotate': 0.0,
        'tex_translate': (0.0, 0.0),
    }
    p = off + 44
    end = off + size
    srt = {}
    if not flags & 0x0002 and p + 8 <= end:
        srt['tex_scale'] = (_s32(d, p) / 4096.0, _s32(d, p + 4) / 4096.0)
        p += 8
    if not flags & 0x0004 and p + 4 <= end:
        sin = _s16(d, p) / 4096.0
        cos = _s16(d, p + 2) / 4096.0
        srt['tex_rotate'] = math.degrees(math.atan2(sin, cos))
        p += 4
    if not flags & 0x0008 and p + 8 <= end:
        srt['tex_translate'] = (_s32(d, p) / 4096.0, _s32(d, p + 4) / 4096.0)
        p += 8
    if flags & 0x0001:
        material.update(srt)
    if flags & 0x2000 and p + 64 <= end:
        values = [_s32(d, p + 4 * k) / 4096.0 for k in range(16)]
        material['tex_effect_mtx'] = [(values[4 * r], values[4 * r + 1])
                                      for r in range(4)]
    return material


_SBC_SIZES = {0x00: 0, 0x02: 2, 0x03: 1, 0x04: 1, 0x05: 1, 0x0A: 8,
              0x0B: 0, 0x0C: 2, 0x0D: 2}


def read_sbc(d, start, end):
    commands = []
    p = start
    while p < end:
        op = d[p]
        p += 1
        base = op & 0x1F
        if base == 0x01:
            break
        if base == 0x09:
            size = 2 + 3 * d[p + 1]
        elif base in (0x06, 0x07, 0x08):
            size = (3 if base == 0x06 else 1) + bool(op & 0x20) + bool(op & 0x40)
        elif base in _SBC_SIZES:
            size = _SBC_SIZES[base]
        else:
            raise NsbmdError("Unknown render command 0x%02X, the model data "
                             "may be corrupted." % op)
        commands.append((op, list(d[p:p + size])))
        p += size
    return commands


def read_model(d, model_off, name):
    nodes_off = model_off + 64
    node_names, node_entries = read_dict(d, nodes_off)
    nodes = []
    for i, (node_name, entry) in enumerate(zip(node_names, node_entries)):
        node = read_node(d, nodes_off + _u32(d, entry))
        node['name'] = node_name or 'node_%d' % i
        nodes.append(node)

    mat_off = model_off + _u32(d, model_off + 8)
    mat_names, mat_entries = read_dict(d, mat_off + 4)
    materials = [read_material(d, mat_name or 'material_%d' % i,
                               mat_off + _u32(d, entry))
                 for i, (mat_name, entry) in enumerate(zip(mat_names, mat_entries))]

    shp_off = model_off + _u32(d, model_off + 12)
    shp_names, shp_entries = read_dict(d, shp_off)
    shapes = []
    for i, (shp_name, entry) in enumerate(zip(shp_names, shp_entries)):
        so = shp_off + _u32(d, entry)
        shapes.append({'name': shp_name or 'shape_%d' % i,
                       'flags': _u32(d, so + 4),
                       'dl_off': so + _u32(d, so + 8),
                       'dl_size': _u32(d, so + 12)})

    model_end = model_off + _u32(d, model_off)
    sbc = read_sbc(d, model_off + _u32(d, model_off + 4), model_end)

    return {
        'name': name,
        'scaling_rule': d[model_off + 21],
        'pos_scale': _s32(d, model_off + 28) / 4096.0,
        'vertex_count': _u16(d, model_off + 36),
        'triangle_count': _u16(d, model_off + 40),
        'quad_count': _u16(d, model_off + 42),
        'nodes': nodes,
        'materials': materials,
        'shapes': shapes,
        'sbc': sbc,
    }


def read_models(d):
    try:
        if _u16(d, 0x0E) < 1:
            raise NsbmdError("The file contains no data blocks.")
        blk = _u32(d, 0x10)
        if d[blk:blk + 4] != b'MDL0':
            raise NsbmdError("The file has no model block (MDL0).")
        names, entries = read_dict(d, blk + 8)
        return [read_model(d, blk + _u32(d, entry), name or 'model_%d' % i)
                for i, (name, entry) in enumerate(zip(names, entries))]
    except (struct.error, IndexError):
        raise NsbmdError("The model data is truncated or corrupted.")


def run_sbc(model):
    nodes = model['nodes']
    identity = _identity()
    cur = identity
    cur_bind = None
    stack = {}
    stack_bind = {}
    world = {}
    parent = {}
    scales = {}
    draws = []
    material = None
    for op, args in model['sbc']:
        base = op & 0x1F
        if base == 0x03:
            cur = stack.get(args[0], identity)
            cur_bind = stack_bind.get(args[0])
        elif base == 0x04:
            material = args[0]
        elif base == 0x05:
            draws.append({'shape': args[0], 'material': material,
                          'matrix': cur, 'binding': cur_bind,
                          'stack': dict(stack), 'stack_bind': dict(stack_bind)})
        elif base in (0x06, 0x07, 0x08):
            i = 3 if base == 0x06 else 1
            dst = src = None
            if op & 0x20:
                dst = args[i]
                i += 1
            if op & 0x40:
                src = args[i]
            if src is not None:
                cur = stack.get(src, identity)
                cur_bind = stack_bind.get(src)
            node_id = args[0]
            if base == 0x06 and node_id < len(nodes):
                parent_id = args[1]
                parent_scale = None
                if (model['scaling_rule'] == 1 and args[2] & 1
                        and parent_id != node_id):
                    parent_scale = scales.get(parent_id)
                cur = _mul(cur, node_local_matrix(nodes[node_id], parent_scale))
                cur_bind = ((node_id, 1.0),)
                scales[node_id] = nodes[node_id]['scale']
                if node_id not in world:
                    world[node_id] = cur
                    if parent_id != node_id and parent_id < len(nodes):
                        parent[node_id] = parent_id
            if dst is not None:
                stack[dst] = cur
                stack_bind[dst] = cur_bind
        elif base == 0x09:
            dst = args[0]
            terms = [(args[2 + 3 * k], args[3 + 3 * k], args[4 + 3 * k])
                     for k in range(args[1])]
            total = float(sum(w for _, _, w in terms)) or 1.0
            mix = [[0.0] * 4 for _ in range(4)]
            binding = []
            for slot, node_id, weight in terms:
                inv = _affine_inverse(world.get(node_id, identity))
                m = _mul(stack.get(slot, identity), inv)
                for r in range(4):
                    for c in range(4):
                        mix[r][c] += m[r][c] * weight / total
                binding.append((node_id, weight / total))
            stack[dst] = mix
            stack_bind[dst] = tuple(binding)
            cur = mix
            cur_bind = tuple(binding)
    for i, node in enumerate(nodes):
        if i not in world:
            world[i] = node_local_matrix(node)
    return world, parent, draws


_GX_PARAMS = {
    0x00: 0, 0x10: 1, 0x11: 0, 0x12: 1, 0x13: 1, 0x14: 1, 0x15: 0,
    0x16: 16, 0x17: 12, 0x18: 16, 0x19: 12, 0x1A: 9, 0x1B: 3, 0x1C: 3,
    0x20: 1, 0x21: 1, 0x22: 1, 0x23: 2, 0x24: 1, 0x25: 1, 0x26: 1,
    0x27: 1, 0x28: 1, 0x29: 1, 0x2A: 1, 0x2B: 1, 0x30: 1, 0x31: 1,
    0x32: 1, 0x33: 1, 0x34: 32, 0x40: 1, 0x41: 0, 0x50: 1, 0x60: 1,
    0x70: 3, 0x71: 2, 0x72: 1,
}

_PRIMITIVE_TYPES = ('triangles', 'quads', 'triangle_strip', 'quad_strip')


def decode_shape(d, shape, draw, pos_scale):
    identity = _identity()
    stack = draw['stack']
    stack_bind = draw['stack_bind']
    mtx = draw['matrix']
    nmtx = _normal_matrix(mtx)
    binding = draw['binding']
    pos = [0.0, 0.0, 0.0]
    normal = None
    uv = None
    color = None
    primitives = []
    prim = None
    p = shape['dl_off']
    end = p + shape['dl_size']
    while p + 4 <= end:
        opcodes = d[p:p + 4]
        p += 4
        for op in opcodes:
            count = _GX_PARAMS.get(op)
            if count is None:
                raise NsbmdError("Shape '%s' contains an unknown display list "
                                 "command 0x%02X." % (shape['name'], op))
            if p + 4 * count > end:
                raise NsbmdError("Shape '%s' has a truncated display list."
                                 % shape['name'])
            args = struct.unpack_from('<%dI' % count, d, p)
            p += 4 * count
            if op == 0x14:
                slot = args[0] & 0x1F
                mtx = stack.get(slot, identity)
                nmtx = _normal_matrix(mtx)
                binding = stack_bind.get(slot)
            elif op == 0x15:
                mtx = identity
                nmtx = _normal_matrix(mtx)
            elif op == 0x20:
                color = _color5(args[0])
            elif op == 0x21:
                normal = (_signed(args[0], 10) / 512.0,
                          _signed(args[0] >> 10, 10) / 512.0,
                          _signed(args[0] >> 20, 10) / 512.0)
            elif op == 0x22:
                uv = (_signed(args[0], 16) / 16.0,
                      _signed(args[0] >> 16, 16) / 16.0)
            elif op == 0x40:
                if prim is not None and prim['vertices']:
                    primitives.append(prim)
                prim = {'type': _PRIMITIVE_TYPES[args[0] & 3], 'vertices': []}
            elif op == 0x41:
                if prim is not None and prim['vertices']:
                    primitives.append(prim)
                prim = None
            elif 0x23 <= op <= 0x28:
                a = args[0]
                if op == 0x23:
                    pos = [_signed(a, 16) / 4096.0, _signed(a >> 16, 16) / 4096.0,
                           _signed(args[1], 16) / 4096.0]
                elif op == 0x24:
                    pos = [_signed(a, 10) / 64.0, _signed(a >> 10, 10) / 64.0,
                           _signed(a >> 20, 10) / 64.0]
                elif op == 0x25:
                    pos = [_signed(a, 16) / 4096.0, _signed(a >> 16, 16) / 4096.0,
                           pos[2]]
                elif op == 0x26:
                    pos = [_signed(a, 16) / 4096.0, pos[1],
                           _signed(a >> 16, 16) / 4096.0]
                elif op == 0x27:
                    pos = [pos[0], _signed(a, 16) / 4096.0,
                           _signed(a >> 16, 16) / 4096.0]
                else:
                    pos = [pos[0] + _signed(a, 10) / 4096.0,
                           pos[1] + _signed(a >> 10, 10) / 4096.0,
                           pos[2] + _signed(a >> 20, 10) / 4096.0]
                if prim is not None:
                    scaled = (pos[0] * pos_scale, pos[1] * pos_scale,
                              pos[2] * pos_scale)
                    prim['vertices'].append({
                        'pos': _apply(mtx, scaled),
                        'normal': (_apply_normal(nmtx, normal)
                                   if normal is not None else None),
                        'uv': uv,
                        'color': color,
                        'binding': binding,
                    })
    if prim is not None and prim['vertices']:
        primitives.append(prim)
    return primitives


def primitive_faces(prim):
    n = len(prim['vertices'])
    kind = prim['type']
    if kind == 'triangles':
        return [(i, i + 1, i + 2) for i in range(0, n - 2, 3)]
    if kind == 'quads':
        return [(i, i + 1, i + 2, i + 3) for i in range(0, n - 3, 4)]
    if kind == 'triangle_strip':
        return [(i, i + 1, i + 2) if i % 2 == 0 else (i + 1, i, i + 2)
                for i in range(n - 2)]
    return [(i, i + 1, i + 3, i + 2) for i in range(0, n - 3, 2)]


def build_geometry(d, model):
    world, parent, draws = run_sbc(model)
    shapes = model['shapes']
    materials = model['materials']
    vertices = []
    vertex_binding = []
    lookup = {}
    faces = []
    face_material = []
    corner_uv = []
    corner_color = []
    corner_normal = []
    vertex_commands = 0
    triangles = 0
    quads = 0
    skipped = 0
    colored_materials = set()
    for draw in draws:
        if draw['shape'] >= len(shapes):
            continue
        mat_id = draw['material']
        mat = materials[mat_id] if mat_id is not None and mat_id < len(materials) else None
        tex_w = mat['tex_width'] if mat else 0
        tex_h = mat['tex_height'] if mat else 0
        for prim in decode_shape(d, shapes[draw['shape']], draw, model['pos_scale']):
            verts = prim['vertices']
            vertex_commands += len(verts)
            prim_faces = primitive_faces(prim)
            for face in prim_faces:
                if len(face) == 3:
                    triangles += 1
                else:
                    quads += 1
                indices = []
                for i in face:
                    v = verts[i]
                    key = (round(v['pos'][0], 5), round(v['pos'][1], 5),
                           round(v['pos'][2], 5), v['binding'])
                    index = lookup.get(key)
                    if index is None:
                        index = len(vertices)
                        lookup[key] = index
                        vertices.append(v['pos'])
                        vertex_binding.append(v['binding'])
                    indices.append(index)
                if len(set(indices)) != len(indices):
                    skipped += 1
                    continue
                faces.append(indices)
                face_material.append(mat_id)
                for i in face:
                    v = verts[i]
                    if v['uv'] is not None and tex_w and tex_h:
                        corner_uv.append((v['uv'][0] / tex_w,
                                          1.0 - v['uv'][1] / tex_h))
                    else:
                        corner_uv.append(None)
                    corner_color.append(v['color'])
                    corner_normal.append(v['normal'])
                    if v['color'] is not None and mat_id is not None:
                        colored_materials.add(mat_id)
    return {
        'world': world,
        'parent': parent,
        'vertices': vertices,
        'vertex_binding': vertex_binding,
        'faces': faces,
        'face_material': face_material,
        'corner_uv': corner_uv,
        'corner_color': corner_color,
        'corner_normal': corner_normal,
        'colored_materials': colored_materials,
        'stats': {'vertex_commands': vertex_commands, 'triangles': triangles,
                  'quads': quads, 'degenerate': skipped},
    }


DS_TO_BLENDER = ((1.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0, -1.0, 0.0),
                 (0.0, 1.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 1.0))


def _s2lin(x):
    if x <= 0.04045:
        return x / 12.92
    return ((x + 0.055) / 1.055) ** 2.4


def _material_color(c):
    return tuple(_s2lin(v / 31.0) for v in c)


def apply_material_settings(material, mat, index, vertex_colored):
    material.is_nns = True
    material.use_nodes = True
    material.blend_method = 'CLIP'
    material.nns_material_id = index
    material.nns_color_type = 'vertex' if vertex_colored else 'solid'
    material.nns_diffuse = _material_color(mat['diffuse'])
    material.nns_ambient = _material_color(mat['ambient'])
    material.nns_specular = _material_color(mat['specular'])
    material.nns_emission = _material_color(mat['emission'])
    material.nns_use_srst = mat['use_srst']
    material.nns_light0 = mat['lights'][0]
    material.nns_light1 = mat['lights'][1]
    material.nns_light2 = mat['lights'][2]
    material.nns_light3 = mat['lights'][3]
    material.nns_polygon_mode = mat['polygon_mode']
    if mat['front'] and mat['back']:
        material.nns_display_face = 'both'
    elif mat['back']:
        material.nns_display_face = 'back'
    else:
        material.nns_display_face = 'front'
    material.nns_update_depth_buffer = mat['update_depth']
    material.nns_far_clipping = mat['far_clip']
    material.nns_render_1_pixel = mat['one_dot']
    material.nns_depth_test = mat['depth_equal']
    material.nns_fog = mat['fog']
    material.nns_alpha = mat['alpha']
    material.nns_polygonid = mat['polygon_id']
    material.nns_wireframe = mat['wireframe']
    material.nns_tex_tiling_u = mat['tiling_u']
    material.nns_tex_tiling_v = mat['tiling_v']
    material.nns_tex_gen_mode = mat['tex_gen_mode']
    material.nns_tex_scale = mat['tex_scale']
    material.nns_tex_rotate = mat['tex_rotate']
    material.nns_tex_translate = mat['tex_translate']
    effect = mat.get('tex_effect_mtx')
    if effect is not None:
        material.nns_tex_effect_mtx_0 = effect[0]
        material.nns_tex_effect_mtx_1 = effect[1]
        material.nns_tex_effect_mtx_2 = effect[2]
        material.nns_tex_effect_mtx_3 = effect[3]
    material.nns_mat_type = compute_mat_type(material)
    generate_nodes(material)


def build_armature(context, model, geometry, convert_axes, bone_length,
                   skip_root):
    nodes = model['nodes']
    world = geometry['world']
    parent = geometry['parent']
    skipped = {i for i in range(len(nodes)) if skip_root and i not in parent}
    if len(skipped) == len(nodes):
        return None, set()
    conv = Matrix(DS_TO_BLENDER)
    arm_data = bpy.data.armatures.new(model['name'])
    arm_obj = bpy.data.objects.new(model['name'], arm_data)
    context.collection.objects.link(arm_obj)
    context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bones = {}
    for i, node in enumerate(nodes):
        if i in skipped:
            continue
        bone = arm_data.edit_bones.new(node['name'])
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, bone_length, 0.0)
        bones[i] = bone
    for i, bone in bones.items():
        p = parent.get(i)
        seen = set()
        while p is not None and p in skipped and p not in seen:
            seen.add(p)
            p = parent.get(p)
        if p is not None and p in bones:
            bone.parent = bones[p]
        m = Matrix(world[i])
        if convert_axes:
            m = conv @ m
        bone.matrix = m
        bone.length = bone_length
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj, set(bones)


def _set_corner_colors(attribute_data, colors):
    for item, c in zip(attribute_data, colors):
        value = (c[0] / 31.0, c[1] / 31.0, c[2] / 31.0, 1.0)
        if hasattr(item, 'color_srgb'):
            item.color_srgb = value
        else:
            item.color = value


def build_mesh(context, model, geometry, arm_obj, bone_nodes, convert_axes):
    name = model['name']
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name + "_mesh" if arm_obj else name, mesh)
    context.collection.objects.link(obj)

    vertices = geometry['vertices']
    if convert_axes:
        vertices = [(v[0], -v[2], v[1]) for v in vertices]
    mesh.from_pydata(vertices, [], geometry['faces'])

    used = sorted({m for m in geometry['face_material'] if m is not None})
    slot_of = {}
    for mat_id in used:
        mat = model['materials'][mat_id]
        material = bpy.data.materials.new(mat['name'])
        apply_material_settings(material, mat, mat_id,
                                mat_id in geometry['colored_materials'])
        slot_of[mat_id] = len(mesh.materials)
        mesh.materials.append(material)
    mesh.polygons.foreach_set(
        'material_index',
        [slot_of.get(m, 0) for m in geometry['face_material']])

    corner_uv = geometry['corner_uv']
    if any(uv is not None for uv in corner_uv):
        uv_layer = mesh.uv_layers.new(name="UVMap")
        flat = []
        for uv in corner_uv:
            flat.extend(uv if uv is not None else (0.0, 0.0))
        uv_layer.data.foreach_set('uv', flat)

    corner_color = geometry['corner_color']
    if any(c is not None for c in corner_color):
        colors = [c if c is not None else (31, 31, 31) for c in corner_color]
        if bpy.app.version >= (3, 2, 0):
            attribute = mesh.color_attributes.new(
                name="Col", type='BYTE_COLOR', domain='CORNER')
            _set_corner_colors(attribute.data, colors)
            mesh.color_attributes.active_color = attribute
        else:
            layer = mesh.vertex_colors.new(name="Col")
            _set_corner_colors(layer.data, colors)

    corner_normal = geometry['corner_normal']
    if any(n is not None for n in corner_normal):
        normals = []
        for n in corner_normal:
            if n is None:
                normals.append((0.0, 0.0, 0.0))
            elif convert_axes:
                normals.append((n[0], -n[2], n[1]))
            else:
                normals.append(n)
        mesh.polygons.foreach_set('use_smooth', [True] * len(mesh.polygons))
        if bpy.app.version < (4, 1, 0):
            mesh.use_auto_smooth = True
        mesh.normals_split_custom_set(normals)

    mesh.update()

    if arm_obj is not None:
        groups = {}
        for i, binding in enumerate(geometry['vertex_binding']):
            if not binding:
                continue
            for node_id, weight in binding:
                if node_id not in bone_nodes:
                    continue
                bone_name = model['nodes'][node_id]['name']
                group = groups.get(bone_name)
                if group is None:
                    group = obj.vertex_groups.new(name=bone_name)
                    groups[bone_name] = group
                group.add([i], weight, 'REPLACE')
        modifier = obj.modifiers.new(name="Armature", type='ARMATURE')
        modifier.object = arm_obj
        obj.parent = arm_obj

    return obj, len(used)


_model_items_cache = {'path': None, 'items': [('0', "Model 0", "")]}


def _model_items(self, context):
    path = self.filepath
    if _model_items_cache['path'] != path:
        items = []
        try:
            for i, model in enumerate(read_models(load_file(path))):
                items.append((str(i), "%s (%d nodes)" % (model['name'],
                                                         len(model['nodes'])), ""))
        except Exception:
            items = []
        _model_items_cache['path'] = path
        _model_items_cache['items'] = items or [('0', "Model 0", "")]
    return _model_items_cache['items']


class NTR_OT_import_nsbmd(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.nns_nsbmd"
    bl_label = "Import NSBMD Model"
    bl_description = ("Import the geometry, armature and NNS material "
                      "settings of an NSBMD model. Textures are left "
                      "unassigned")
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".nsbmd"
    filter_glob: StringProperty(default="*.nsbmd;*.bmd;*.bin",
                                options={'HIDDEN'})

    model_index: EnumProperty(
        name="Model",
        description="Which model in the file to import",
        items=_model_items)

    import_mesh: BoolProperty(
        name="Import mesh and materials",
        description=("Build the geometry, weighted to the armature, with "
                     "NNS material settings matching the file"),
        default=True)

    convert_axes: BoolProperty(
        name="Convert to Blender axes",
        description=("Stand the model up in Blender's Z-up space. Export "
                     "it with the default Root rotation (-90 on X) to get "
                     "the same orientation back. Leave this off to keep the "
                     "file's own Y-up space, then export with Root rotation "
                     "set to zero"),
        default=False)

    skip_root: BoolProperty(
        name="Skip root node",
        description=("Don't create a bone for the file's top level node. The "
                     "exporter always adds its own root node, so importing "
                     "this one as a bone would give the exported model one "
                     "node too many"),
        default=True)

    bone_length: FloatProperty(
        name="Bone length",
        description="Display length of the created bones",
        default=0.05,
        min=0.0001)

    def execute(self, context):
        try:
            data = load_file(self.filepath)
            models = read_models(data)
            if not models:
                raise NsbmdError("The file doesn't contain any model.")
            try:
                index = int(self.model_index)
            except (TypeError, ValueError):
                index = 0
            model = models[index if index < len(models) else 0]
            geometry = build_geometry(data, model)
        except NsbmdError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except (OSError, struct.error, IndexError) as e:
            self.report({'ERROR'}, "Couldn't read this file: %s" % e)
            return {'CANCELLED'}

        if context.object is not None and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for obj in context.selected_objects:
            obj.select_set(False)

        arm_obj, bone_nodes = build_armature(
            context, model, geometry, self.convert_axes, self.bone_length,
            self.skip_root)
        message = "Imported '%s'" % model['name']
        if arm_obj is not None:
            message += ", %d bones" % len(arm_obj.data.bones)

        if self.import_mesh and geometry['faces']:
            mesh_obj, material_count = build_mesh(
                context, model, geometry, arm_obj, bone_nodes,
                self.convert_axes)
            mesh_obj.select_set(True)
            context.view_layer.objects.active = mesh_obj
            message += ", %d vertices, %d faces, %d materials" % (
                len(geometry['vertices']), len(geometry['faces']),
                material_count)
        elif arm_obj is not None:
            context.view_layer.objects.active = arm_obj

        self.report({'INFO'}, message)
        return {'FINISHED'}


def menu_func_import_nsbmd(self, context):
    self.layout.operator(NTR_OT_import_nsbmd.bl_idname,
                         text="NNS Model (.nsbmd)")


def import_register():
    safe_register_class(NTR_OT_import_nsbmd)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_nsbmd)


def import_unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_nsbmd)
    bpy.utils.unregister_class(NTR_OT_import_nsbmd)
