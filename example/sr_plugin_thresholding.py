import getopt
import sys
import numpy as np
import slicerecon


def set_threshold(argv):
    thrs = 0.5 # Default value of the threshold
    try:
        opts, args = getopt.getopt(argv,"t:",["threshold="])
    except getopt.GetoptError:
        print('sr_plugin_thresholding.py -t <threshold>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-t", "--threshold"):
            thrs = np.float64(arg)
        else:
            print('sr_plugin_thresholding.py -t <threshold>')
            sys.exit()
    print("Threshold set to {}.".format(thrs))
    return thrs

threshold = set_threshold(sys.argv[1:])

def callback(shape, xs, idx):
    xs = np.array(xs).reshape(shape)
    print(xs.max(), xs.min())
    if xs.max() - xs.min() != 0:
        xs = (xs / (xs.max() - xs.min()))

    print("callback called", shape)
    xs[xs > threshold] = 10.0
    xs[xs <= threshold] = 0.0
    xs *= -1.0
    xs += 10.0
    print(xs.max(),xs.min())
    return [shape, xs.ravel().tolist()]

p = slicerecon.plugin("tcp://*:5652", "tcp://localhost:5555")
p.set_slice_callback(callback)

p.listen()
