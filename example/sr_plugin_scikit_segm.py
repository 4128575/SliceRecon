import getopt
import sys
import numpy as np
import slicerecon
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from timeit import default_timer as timer

def set_threshold(argv):
    thrs1 = 0.35 # Default value of the threshold
    thrs2 = 0.15
    try:
        opts, args = getopt.getopt(argv,"a:b:",["threshold_1=","threshold_2="])
    except getopt.GetoptError:
        print('sr_plugin_thresholding.py -a <threshold_1> -b <threshold_2>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-a", "--threshold_1"):
            thrs1 = np.float64(arg)
        elif opt in ("-b", "--threshold_2"):
            thrs2 = np.float64(arg)
        else:
            print('sr_plugin_thresholding.py -a <threshold_1> -b <threshold_2>')
            sys.exit()
    print("Thresholds set to {} and {}.".format(thrs1, thrs2))
    return thrs1, thrs2

threshold1, threshold2 = set_threshold(sys.argv[1:])

def supervised_thresholding(xs):
    xs[xs > threshold1] = 1.0
    xs[xs <= threshold2] = 1.0
    xs[xs <= threshold1] = -1.0
    xs *= -1.0    
    return xs

def unsupervised_thresholding(xs):
    #xs_thr = filters.threshold_mean(xs) # TV 2487
    #xs_thr = filters.threshold_local(xs, block_size=25) # TV 41228
    xs_thr = filters.threshold_otsu(xs) # TV 2375
    #xs_thr = filters.threshold_li(xs) # TV 2459
    #xs_thr = filters.threshold_niblack(xs, window_size=25) # TV 33100 
    #xs_thr = filters.threshold_sauvola(xs, window_size=25) # TV 22346
    #xs_thr = filters.threshold_triangle(xs) # TV 2917
    #xs_thr = filters.threshold_minimum(xs, max_iter=1000) #TV 2277, TODO: Segfault
    return_im = (xs < xs_thr).astype(np.float32)
    return return_im


def unsupervised_chan_vese(xs, mu=0.01):
    image_slic = seg.chan_vese(xs, mu=mu, max_iter=200, init_level_set="small disk")
    return image_slic*-1.0

def unsupervised_morph_chan_vese(xs):
    image_slic = seg.morphological_chan_vese(xs, 100, init_level_set="checkerboard", smoothing=1)
    return image_slic

def calc_tv_norm(xs):
    pixel_dif_ver = xs[1:, :] - xs[:-1, :]
    pixel_dif_hor = xs[:, 1:] - xs[:, :-1]
    tot_var = np.sum(np.abs(pixel_dif_ver)) + np.sum(np.abs(pixel_dif_hor))
    return tot_var

def unsupervised_inv_gaussian(xs):
    image_slic = seg.inverse_gaussian_gradient(xs)
    return image_slic*-1.0


def callback(shape, xs, idx):
    print("callback called", shape)
    #start = timer()
    xs = np.array(xs).reshape(shape)
    if xs.shape[0] < 512:
        return [shape, xs.ravel().tolist()]
    if xs.max() - xs.min() != 0:
        xs = (xs / (xs.max() - xs.min()))
    xs -= xs.min()
    
    #ys = supervised_thresholding(xs)
    ys = unsupervised_thresholding(xs)
    #ys = unsupervised_chan_vese(xs, 0.01)
    #ys = unsupervised_inv_gaussian(xs)
    ys = (ys / (ys.max() - ys.min()))
    ys -= ys.min()
    
    tvnorm = calc_tv_norm(ys)
    l2norm = np.linalg.norm(xs-ys)
    print("TV norm:", tvnorm, "L2 norm:", l2norm)
    #end = timer()
    #print("\n * Time to process slice: %.1f ms" % (end - start)*1000)
    xs = ys
    return [shape, xs.ravel().tolist()]

p = slicerecon.plugin("tcp://*:5652", "tcp://localhost:5555")
p.set_slice_callback(callback)

p.listen()
