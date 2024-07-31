import numpy as np
import pickle
import json


def save_dict(outdir, filename, output):

    fname = outdir + "/" + filename

    np.save(fname + ".npy", output)

    with open(fname + ".txt", "w") as handle:
        print(output, file=handle)

    with open(fname + ".pickle", "wb") as handle:
        """
        Using protocol=pickle.HIGHEST_PROTOCOL may cause
        issues with older Python versions.
        """

        pickle.dump(output, handle, protocol=4)


