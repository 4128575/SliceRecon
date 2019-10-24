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
    -pt --path, set path to trained network .../*.pytorch
    """
    ### Default values ###
    c_in = 1
    num_labels = 2
    depth = 50
    width = 1
    dilations = [1,2,3,4,5,6,7,8,9,10]
    parallel=False  #multiple GPUs or not
    batch_size = 1
    net_path = "/export/scratch2/schoonho/Planmeca/
        trained_networks/msd_axial_jordi.pytorch"
    try:
        opts, args = getopt.getopt(argv,"hc:n:d:w:di:p:bs:pt:",["help=","cin=",
            "num_labels=","depth=","width=","dilations=","parallel=",
            "batch_size=","path="])
    except getopt.GetoptError:
        print('sr_plugin_thresholding.py -c <channels_in> -n <number_of_labels> -d <depth>
                -w <width> -di <list_dilations> -p <parallel> -bs <batch_size>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('sr_plugin_thresholding.py -c <channels_in> -n <number_of_labels> 
                    -d <depth> -w <width> -di <list_dilations> 
                    -p <parallel> -bs <batch_size>')
            sys.exit()
        elif opt in ("-c", "--cin"):
            c_in = arg
        elif opt in ("-n", "--num_labels"):
            num_labels = arg
        elif opt in ("-d", "--depth"):
            depth = arg
        elif opt in ("-w", "--width"):
            width = arg
        elif opt in ("-di", "--dilations"):
            dilations = arg
        elif opt in ("-p", "--parallel"):
            parallel = arg
        elif opt in ("-bs", "--batch_size"):
            batch_size = arg
        elif opt in ("-pt", "--path"):
            net_path = arg
        else:
            print('sr_plugin_thresholding.py -c <channels_in> -n <number_of_labels> 
                    -d <depth> -w <width> -di <list_dilations> 
                    -p <parallel> -bs <batch_size>')
            sys.exit()
    return c_in, num_labels, depth, width, dilations, parallel, batch_size, net_path

c_in, num_labels, depth, width, dilations, parallel, batch_size, net_path =
    set_config(sys.argv[1:])

def load_network(net_path):
    model = mp.MSDSegmentationModel(c_in, num_labels, depth,
            width, dilations=dilations, parallel=parallel)
    model.load(net_path)
    return model

model = load_network(net_path)

def callback(shape, xs, idx):
    xs = np.array(xs).reshape(shape)
    print("callback called", shape)
    output_slice = np.zeros(shape=shape)
    model.forward(xs)
    #TODO: Generalize for arbitrary segmentation with arguments given in command line
    output_slice = model.output.detach().cpu()[0][1]
    #TODO: Threshold segmentation at 0.5, also with cmd argument
    #output_slice[output_slice <= threshold] = 0.0
    #output_slice[output_slice > threshold] = 10.0
    return [shape, output_slice.ravel().tolist()]

p = slicerecon.plugin("tcp://*:5652", "tcp://localhost:5555")
p.set_slice_callback(callback)

p.listen()
