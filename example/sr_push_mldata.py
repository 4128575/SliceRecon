import tomop
import flexdata as flex
import numpy as np
import scipy.misc
import astra
import argparse
import tifffile

parser = argparse.ArgumentParser(
    description='Push a FleX-ray data set to Slicerecon.')
parser.add_argument('path', metavar='path', help='path to the data')
parser.add_argument(
    '--sample',
    type=int,
    default=1,
    help='the binning to use on the detector, and how many projections to skip'
)
parser.add_argument(
    '--host', default="localhost", help='the projection server host')
parser.add_argument(
    '--port', type=int, default=5558, help='the projection server port')

parser.add_argument(
    '--skipgeometry',
    action='store_true',
    help='assume the geometry packet is already sent')
parser.add_argument(
    '--linearize', action='store_true', help='whether data is linear')
args = parser.parse_args()

path = args.path
proj = tifffile.imread(path)
n = 2*proj.shape[0]//3
print(n)
factor = 3*n//2
rows = proj.shape[0]
cols = proj.shape[2]
nr_angles = proj.shape[1]
t = np.linspace(0.0, 360.0, num=nr_angles, endpoint=False)
angles = np.array(t)*np.pi/180
proj_count = len(angles)

print(proj.min(),proj.max())
dark = np.full((rows,cols), 0).astype(np.float32)
flat = np.full((rows,cols), 1).astype(np.float32)
proj = np.exp(-proj)
print(proj.shape)

proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, factor, factor, angles, 1000, 0)
vol_geom = astra.create_vol_geom(n,n,n)
proj_geom = astra.functions.geom_2vec(proj_geom)

pub = tomop.publisher(args.host, args.port)

# send astra geometry
print('proj geom vector count', len(proj_geom['Vectors']))

packet_vol_geom = tomop.geometry_specification_packet(0, [
    vol_geom['option']['WindowMinX'], vol_geom['option']['WindowMinY'],
    vol_geom['option']['WindowMinZ']
], [
    vol_geom['option']['WindowMaxX'], vol_geom['option']['WindowMaxY'],
    vol_geom['option']['WindowMaxZ']
])
if not args.skipgeometry:
    pub.send(packet_vol_geom)


packet_scan_settings = tomop.scan_settings_packet(0, 0, 0)
if not args.skipgeometry:
    pub.send(packet_scan_settings)

#print("Sending scan data (linear: ", not args.linearize, ")")
#pub.send(tomop.scan_settings_packet(0, 0, 0, not args.linearize))

packet_geometry = tomop.cone_vec_geometry_packet(
    0, rows, cols, proj_count, proj_geom['Vectors'].flatten())
if not args.skipgeometry:
    pub.send(packet_geometry)

#packet_dark = tomop.projection_packet(
#    0, 0, [rows, cols], np.ascontiguousarray(dark.flatten()))
#pub.send(packet_dark)

#packet_light = tomop.projection_packet(
#    1, 0, [rows, cols], np.ascontiguousarray(flat.flatten()))
#pub.send(packet_light)

proj = np.swapaxes(proj, 0, 1)
for i in np.arange(0, proj_count):
    packet_proj = tomop.projection_packet(
        2, i, [rows, cols], np.ascontiguousarray(proj[i].flatten()))
    pub.send(packet_proj)
