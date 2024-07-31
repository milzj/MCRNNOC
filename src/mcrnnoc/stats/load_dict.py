import pickle

def load_dict(outdir, filename):

    fname = outdir + "/" + filename

    with open(fname + ".pickle", "rb") as handle:
        dict = pickle.load(handle)

    return dict
