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

path = args.path
path2 = '/export/scratch2/schoonho/Planmeca/results/original_rec_pht=2_bottom.tif'
vol = (tifffile.imread(path2)).astype(np.float32)
print(vol.shape)
vol = vol / vol.max()

dimX = 1001
dimY = 1001
dimZ = 511

vol_geom = astra.create_vol_geom(dimX, dimY, dimZ)

pub = tomop.publisher(args.host, args.port)

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

packet_vol = tomop.volume_data_packet(0, [dimZ, dimX, dimY],
        np.ascontiguousarray(vol.flatten()))
pub.send(packet_vol)

#for i in np.arange(0, vol.shape[2]):
#    packet_vol = tomop.volume_data_packet(
#            0, [dimX, dimY], np.ascontiguousarray(vol[:,:,i].flatten()))
   # pub.send(packet_vol)
