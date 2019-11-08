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

# Load projections
path = args.path
proj = tifffile.imread(path)

data_dir = '/export/scratch2/schoonho/Planmeca/projection_data_original/'
phantoms = ['Phantom1','Phantom2','Phantom3','Phantom4','Phantom5','Phantom6']
lookup_dict = {
    'top': {
        'Phantom1': '20180219142544/',
        'Phantom2': '20180220104813/',
        'Phantom3': '20180220154819/',
        'Phantom4': '20180221104023/',
        'Phantom5': '20180222141648/',
        'Phantom6': '20180316163313/'},
    'bottom': {
        'Phantom1': '20180219142913/',
        'Phantom2': '20180220105220/',
        'Phantom3': '20180220155206/',
        'Phantom4': '20180221122654/',
        'Phantom5': '20180222142034/',
        'Phantom6': '20180316163629/'}}

def load_geometryfile(pht, half):
    if (pht == 6 and half == 'bottom'):
        raise Exception('Projection data missing!')
    path = data_dir+'Phantom{}/'.format(pht)+lookup_dict[half]['Phantom{}'.format(pht)] +'geometryFile'
    geom_file = np.loadtxt(path, dtype=np.float32)
    return geom_file


# Set geometry parameters
dType = np.uint16 
projX = 570
projY = 576
projZ = 1
nrAngles = 600

dimX = 1001
dimY = 1001
dimZ = 500

# TODO: Dit als flag meegeven
phant = 2
half = 'bottom'
if (phant == 6 and half == 'bottom'):
    raise Exception('Projection data missing!')

geometry_file = load_geometryfile(phant, half)

flat_path = data_dir+'Phantom{}/'.format(phant)+lookup_dict[half]['Phantom{}'.format(phant)] + 'flat.raw'
flat = np.fromfile(flat_path, dtype=dType).reshape(projY, projX)
dark = np.zeros(shape=flat.shape)

vox_d = 200     # Deze getallen staan in metadata
vox_h = 100     # Deze getallen staan in metadata
vox_size = 0.2  # Deze getallen staan in metadata
vox_size_d = vox_d / dimX
vox_size_h = vox_h / dimZ
det_spacing_x = 144.78 / (570.0 * vox_size_d)
det_spacing_y = 146.304 / (576.0 * vox_size_h)

def build_cone_vec(geometryfile):
    vecs = []
    x_angles = geometryfile[:,6] * np.pi/180.0
    y_angles = geometryfile[:,7] * np.pi/180.0
    z_angles = geometryfile[:,8] * np.pi/180.0
    for i in range(geometryfile.shape[0]):
        src_v = np.array([geometryfile[i,0] / vox_size_d,
                geometryfile[i,1] / vox_size_d,
                geometryfile[i,2] / vox_size_h])
        det_v = np.array([geometryfile[i,3] / vox_size_d,
                geometryfile[i,4] / vox_size_d,
                geometryfile[i,5] / vox_size_h])
        detu_v = np.array([np.cos(z_angles[i] + 1*np.pi/2) * det_spacing_x,
                  np.sin(z_angles[i] + 1*np.pi/2) * det_spacing_x,
                  0])
        detv_v = np.array([0,
                  0,
                  det_spacing_y])

        # We have access to rotations of detector around X and Y plane as well
        c = np.cos(x_angles[i])
        s = np.sin(x_angles[i])
        rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        c = np.cos(y_angles[i])
        s = np.sin(y_angles[i])
        ry = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        ra = np.dot(rx,ry)
        detu_v = np.dot(ra,detu_v)
        detv_v = np.dot(ra,detv_v)
        vecs.append(np.array([src_v, det_v, detu_v, detv_v]).flatten())
    vecs = np.array(vecs)
    return vecs

cone_vecs = build_cone_vec(geometry_file)

det_row_count = proj.shape[0]
det_col_count = proj.shape[2]
proj_geom = astra.create_proj_geom('cone_vec',
        det_row_count, det_col_count, cone_vecs)
vol_geom = astra.create_vol_geom(dimX, dimY, dimZ)

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

packet_scan_settings = tomop.scan_settings_packet(0, 1, 1)

if not args.skipgeometry:
    pub.send(packet_scan_settings)

packet_geometry = tomop.cone_vec_geometry_packet(
    0, projY, projX, nrAngles, proj_geom['Vectors'].flatten())
if not args.skipgeometry:
    pub.send(packet_geometry)

for i in np.arange(0, 1):
    packet_dark = tomop.projection_packet(
        0, i, [projY, projX], np.ascontiguousarray(dark.flatten()))
    pub.send(packet_dark)

for i in np.arange(0, 1):
    packet_light = tomop.projection_packet(
        1, i, [projY, projX], np.ascontiguousarray(flat.flatten()))
    pub.send(packet_light)

proj = np.swapaxes(proj, 0, 1)
for i in np.arange(0, nrAngles):
    packet_proj = tomop.projection_packet(
        2, i, [projY, projX], np.ascontiguousarray(proj[i].flatten()))
    pub.send(packet_proj)
