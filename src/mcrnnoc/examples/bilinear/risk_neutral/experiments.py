import numpy as np

class Experiments(object):

  def __init__(self):

    self._experiments = {}

    name = "Monte_Carlo_Rate"
    N_vec = [2**i for i in range(4, 8+1)]
    n_vec = 64*np.ones(len(N_vec), dtype=np.int64)

    self.add_experiment(name, n_vec, N_vec)

    name = "Monte_Carlo_Rate_Fixed_Control"
    N_vec = [2**i for i in range(6, 12+1)]
    n_vec = 64*np.ones(len(N_vec), dtype=np.int64)

    self.add_experiment(name, n_vec, N_vec)

    name = "Monte_Carlo_Rate_Test"
    N_vec = [2**i for i in range(3, 4+1)]
    n_vec = 64*np.ones(len(N_vec), dtype=np.int64)

    self.add_experiment(name, n_vec, N_vec)

    name = "Dimension_Dependence"

    n_vec = [2**i for i in range(3, 7+1)]
    N_vec = 128*np.ones(len(n_vec), dtype=np.int64)

    self.add_experiment(name, n_vec, N_vec)

  def add_experiment(self, name, n_vec, N_vec):

    key = ("n_vec", "N_vec")
    items = list(zip(np.array(n_vec),np.array(N_vec)))

    self._experiments[name] = {key: items}
    self._experiments[name + "_Synthetic"] = {key: items}


  def __call__(self, experiment_name):

    return self._experiments[experiment_name]


if __name__ == "__main__":

  experiments = Experiments()
  print(experiments._experiments)

  experiment_name = "Dimension_Dependence"

  for e in experiments(experiment_name)[("n_vec", "N_vec")]:
    print(e)
    n, N = e
