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
args = parser.parse_args()

sample = args.sample
print("sample", sample)
path = args.path
#path2 = "/export/scratch2/schoonho/RECAST_data/Ribosome_experiments/sinogram_ribosome_cone.tiff"

#dark = flex.io.read_tiffs(path, 'di', sample=sample)
#flat = flex.io.read_tiffs(path, 'io', sample=sample)
#proj = flex.io.read_tiffs(path, 'scan_', sample=sample, skip=sample)
proj = tifffile.imread(path)
print(np.shape(proj))

t = [i for i in range(0,360,1)]
angles = np.array(t)*np.pi/180
rows = 320
cols = 320
proj_count = len(angles)
proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, 320, 320, angles, 1000, 0)
vol_geom = astra.create_vol_geom(320, 320, 320)
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

packet_geometry = tomop.cone_vec_geometry_packet(
    0, rows, cols, proj_count, proj_geom['Vectors'].flatten())
if not args.skipgeometry:
    pub.send(packet_geometry)

proj = np.swapaxes(proj, 0, 1)
for i in np.arange(0, proj_count):
    packet_proj = tomop.projection_packet(
        2, i, [rows, cols], np.ascontiguousarray(proj[i].flatten()))
    pub.send(packet_proj)
