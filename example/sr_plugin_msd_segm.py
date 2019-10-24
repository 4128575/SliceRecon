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
    -di --dilations, list of dilations as [..]
    -p --parallel, bool if want to use multiple GPUs or not
    -bs --batch_size, batch size
    """
    ### Default values ###
    c_in = 1
    num_labels = 2
    depth = 50
    width = 1
    dilations = [1,2,3,4,5,6,7,8,9,10]
    parallel=False  #multiple GPUs or not
    batch_size = 1
    try:
        opts, args = getopt.getopt(argv,"c:n:d:w:di:p:bs:",["cin=","num_labels=",
            "depth=","width=","dilations=","parallel=","batch_size="])
    except getopt.GetoptError:
        print('sr_plugin_thresholding.py -t <threshold>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-t", "--threshold"):
            thrs = np.float64(arg)
        else:
            print('sr_plugin_thresholding.py -t <threshold>')
            sys.exit()
    return c_in, num_labels, depth, width, dilations, parallel, batch_size

c_in, num_labels, depth, width, dilations, parallel, batch_size = set_config(sys.argv[1:])

def callback(shape, xs, idx):
    xs = np.array(xs).reshape(shape)

    print("callback called", shape)
    xs[xs <= threshold] = 0.0
    xs[xs > threshold] = 10.0

    return [shape, xs.ravel().tolist()]

p = slicerecon.plugin("tcp://*:5652", "tcp://localhost:5555")
p.set_slice_callback(callback)

p.listen()
