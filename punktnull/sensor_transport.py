"""Strict sensor-only transport between Isaac Python 3.11 and ROS Humble 3.10.

Length-prefixed JSON plus typed arrays; deliberately no pickle or truth poses.
Uses only the standard library and numpy. Loopback TCP is local simulation IO.
"""
from __future__ import annotations
import json
import socket
import struct
import numpy as np

MAX_PACKET = 16 * 1024 * 1024
ARRAYS = (("xyz_sensor_m", "<f4", 3), ("pointdt_ns", "<u4", 1),
          ("intensity", "<f4", 1), ("line", "u1", 1))

def encode(event):
    kind = event['kind']
    chunks = []
    if kind == 'imu':
        stamp = int(event.get('stamp_ns', event.get('timestamp_ns', -1)))
        m = dict(kind=kind, stamp_ns=stamp)
        for key in ('angular_velocity_rad_s', 'specific_force_m_s2'):
            a = np.asarray(event[key], dtype=float)
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError('Invalid IMU vector: '+key)
            m[key] = a.tolist()
    elif kind == 'cloud':
        stamp = int(event['scanstart_ns'])
        count = len(event['xyz_sensor_m'])
        if not 1 <= count <= 250_000:
            raise ValueError('Invalid cloud count')
        m = dict(kind=kind, stamp_ns=stamp, scanstart_ns=stamp, count=count,
                 scan_end_ns=int(event['scan_end_ns']), complete=bool(event.get('complete', True)))
        for key, dtype, width in ARRAYS:
            a = np.asarray(event[key], dtype=dtype)
            expected = (count, width) if width > 1 else (count,)
            if a.shape != expected or not np.isfinite(a).all():
                raise ValueError('Invalid cloud array: '+key)
            chunks.append(a.tobytes(order='C'))
        dt = np.asarray(event['pointdt_ns'], dtype=np.int64)
        if np.any(np.diff(dt) < 0) or stamp + int(dt.max()) >= m['scan_end_ns']:
            raise ValueError('Point timestamps outside half-open scan')
    else:
        raise ValueError('Unknown sensor packet kind')
    if stamp < 0:
        raise ValueError('Negative timestamp')
    header = json.dumps(m, allow_nan=False, separators=(',', ':')).encode()
    body = struct.pack('!I', len(header)) + header + b''.join(chunks)
    if len(body) > MAX_PACKET:
        raise ValueError('Sensor packet too large')
    return struct.pack('!I', len(body)) + body

def decode(body):
    if len(body) < 4 or len(body) > MAX_PACKET:
        raise ValueError('Invalid packet length')
    n, = struct.unpack('!I', body[:4])
    if not 2 <= n <= min(8192, len(body)-4):
        raise ValueError('Invalid header length')
    m = json.loads(body[4:4+n]); offset = 4+n
    if m.get('kind') == 'cloud':
        count = m['count']
        if not isinstance(count, int) or not 1 <= count <= 250_000:
            raise ValueError('Invalid count')
        for key, dtype, width in ARRAYS:
            size = count * np.dtype(dtype).itemsize * width
            if offset+size > len(body):
                raise ValueError('Truncated cloud')
            a = np.frombuffer(body, dtype=dtype, count=count*width, offset=offset).copy()
            m[key] = a.reshape(count, width) if width > 1 else a
            offset += size
    if offset != len(body):
        raise ValueError('Trailing sensor data')
    encode(m)  # One validation path, including finite values and timestamp order.
    return m

def recv_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(count-len(data))
        if not chunk:
            raise EOFError('Sensor stream closed')
        data.extend(chunk)
    return bytes(data)

def receive(sock):
    size, = struct.unpack('!I', recv_exact(sock, 4))
    if not 4 <= size <= MAX_PACKET:
        raise ValueError('Invalid sensor packet size')
    return decode(recv_exact(sock, size))

def connect(port=16720):
    return socket.create_connection(('127.0.0.1', int(port)), timeout=2)
