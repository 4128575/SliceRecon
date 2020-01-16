import torch
import msd_pytorch as mp
from torch.utils.data import DataLoader
import getopt
import sys
import numpy as np
import slicerecon
from timeit import default_timer as timer

def set_config(argv):
    """
    -c --cin, is # of input channels
    -n --num_labels, is # of labels in segmentation
    -d --depth, is depth of network
    -w --width, is width of network
    -i --dilations, list of dilations as [..]
    -b --batch_size, batch size
    -p --path, set path to trained network .../*.pytorch
    -l --label, segmentation label we wish to see
    -t --threshold, threshold the segmented pixels
    """
    ### Default values ###
    c_in = 1
    num_labels = 2
    depth = 100
    width = 1
    dilations = [1,2,3,4,5,6,7,8,9,10]
    parallel=False  #multiple GPUs or not
    batch_size = 20
    #net_path = "/export/scratch2/schoonho/data/ML_testdata/networks_fdk/networks_n=128/val_no_noise/d=100/msd_regular_d=100_it=3_epoch=49_bs=25_error=0.0012.torch"
    #net_path = "/export/scratch2/schoonho/data/TabletInFluid/trained_networks/bubbledataonly/msd_regular_smoothed_d=100_it=0_epoch=11_bs=20_error=0.0014.torch"
    net_path = "/export/scratch2/schoonho/data/TabletInFluid/trained_networks/bubbledataonly/msd_bubble_random_d=100_it=0_epoch=35_bs=20_error=0.0132.torch"
    #net_path = "/export/scratch2/schoonho/data/TabletInFluid/trained_networks/generatoronly/msd_regular_d=100_it=0_epoch=17_bs=20_error=0.0049.torch"
    label = 1
    threshold = None
    try:
        opts, args = getopt.getopt(argv,"hc:n:d:w:i:b:p:l:t:",["help=","cin=",
            "num_labels=","depth=","width=","dilations=","batch_size=","path=","label=","threshold="])
    except getopt.GetoptError:
        print('sr_plugin_msd_segm.py -c <channels_in> -n <number_of_labels> -d <depth> -w <width> -i <list_dilations> -b <batch_size> -p <path> -l <label> -t <threshold>')
        sys.exit(2)
    for opt, arg in opts:
        print(opt,arg)
        if opt in ("-h", "--help"):
            print('sr_plugin_msd_segm.py -c <channels_in> -n <number_of_labels> -d <depth> -w <width> -i <list_dilations> -b <batch_size> -p <path> -l <label> -t <threshold>')
            sys.exit()
        elif opt in ("-c", "--cin"):
            c_in = int(arg)
        elif opt in ("-n", "--num_labels"):
            num_labels = int(arg)
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-w", "--width"):
            width = int(arg)
        elif opt in ("-i", "--dilations"):
            dilations = arg
        elif opt in ("-b", "--batch_size"):
            batch_size = arg
        elif opt in ("-p", "--path"):
            net_path = arg
        elif opt in ("-l", "--label"):
            label = int(arg)
        elif opt in ("-t", "--threshold"):
            threshold = float(arg)
        else:
            print('sr_plugin_msd_segm.py -c <channels_in> -n <number_of_labels> -d <depth> -w <width> -i <list_dilations> -b <batch_size> -p <path> -l <label> -t <threshold>')
            sys.exit()
    return c_in, num_labels, depth, width, dilations, batch_size, net_path, label, threshold

c_in, num_labels, depth, width, dilations, batch_size, net_path, label, threshold = set_config(sys.argv[1:])

def load_network(net_path):
    print(c_in, num_labels, depth, width, dilations)
    model = mp.MSDSegmentationModel(c_in, num_labels, depth, width, dilations=dilations)
    model.load(net_path)
    print("Loaded network...")
    return model

model = load_network(net_path)

def invert_slice(xs):
    diff = xs.max() + xs.min()
    xs *= -1.0
    xs += diff
    return xs

def callback(shape, xs, idx):
    time = 0
    xs = np.array(xs).reshape(shape)
    if (xs.max() - xs.min()) != 0: # This is only to fix the input range for net
        xs /= 10*(xs.max() - xs.min())
    if xs.shape[0] < 512:
        return [shape, xs.ravel().tolist()]
    
    start = timer()
    xs = torch.Tensor([[xs]]).to('cuda:0')
    model.forward(xs, xs)
    end = timer()
    #preds = model.output.detach().cpu()[0][label].exp().data.numpy()
    _, preds = torch.max(model.output, label)
    preds = preds.cpu().detach().numpy().astype(np.float32)

    time += 1000*(end-start)
    print("*    Time for callback: %.5f ms" % (time))
    
    torch.cuda.empty_cache()
    if threshold is None:
        return [shape, preds.ravel().tolist()]
    else:
        preds[preds <= threshold] = 0.0
        preds[preds > threshold] = 10.0
        return [shape, preds.ravel().tolist()]

p = slicerecon.plugin("tcp://*:5652", "tcp://localhost:5555")
p.set_slice_callback(callback)

p.listen()
