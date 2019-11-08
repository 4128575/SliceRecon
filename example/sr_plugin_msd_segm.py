import torch
import msd_pytorch as mp
from torch.utils.data import DataLoader
import getopt
import sys
import numpy as np
import slicerecon

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
    depth = 50
    width = 1
    dilations = [1,2,3,4,5,6,7,8,9,10]
    parallel=False  #multiple GPUs or not
    batch_size = 1
    net_path = "/export/scratch2/schoonho/ML_testdata/networks/networks_n=128/val/d=50/msd_regular_w=1_it=0_epoch=97.torch"
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
    return model

model = load_network(net_path)

def callback(shape, xs, idx):
    xs = np.array(xs).reshape(shape)
    print(xs.min(), xs.max())
    #xs /= ((-1/3)*(xs.max() - xs.min()))
    print(xs.min(), xs.max())
    if xs.shape[0] < 128:
        return [shape, xs.ravel().tolist()]
    xs = torch.Tensor([[xs]])
    print(xs.size())
    print("callback called", shape)
    model.forward(xs, torch.zeros(xs.size()))
    output_slice = model.output.detach().cpu()[0][label].exp().data.numpy()
    print(output_slice.min(), output_slice.max())
    if threshold is None:
        return [shape, output_slice.ravel().tolist()]
    output_slice[output_slice <= threshold] = 0.0
    output_slice[output_slice > threshold] = 10.0
    return [shape, output_slice.ravel().tolist()]

p = slicerecon.plugin("tcp://*:5652", "tcp://localhost:5555")
p.set_slice_callback(callback)

p.listen()
