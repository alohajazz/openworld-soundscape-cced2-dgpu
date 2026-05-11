import numpy as np
a = np.load("/workspace/cache/16k_mono/1706_20170821_015700_821.npy")
print(f"loaded shape={a.shape} dtype={a.dtype} sec={a.shape[0]/16000:.2f}")
