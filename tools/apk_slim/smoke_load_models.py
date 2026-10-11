import glob
import os
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter

d = sys.argv[1]
for p in sorted(glob.glob(os.path.join(d, "*.tflite"))):
    it = Interpreter(model_path=p)
    it.allocate_tensors()
    i = it.get_input_details()[0]
    o = it.get_output_details()[0]
    x = np.zeros((1,) + tuple(i["shape"][1:]), np.float32)
    it.set_tensor(i["index"], x)
    it.invoke()
    out = it.get_tensor(o["index"])
    print(f"{os.path.basename(p):34s} in {str(i['dtype']).split('.')[-1]:8s} "
          f"{str(list(i['shape'])):16s} out {str(o['dtype']).split('.')[-1]:8s} "
          f"{str(list(o['shape'])):10s} invoke=OK shape={out.shape}")