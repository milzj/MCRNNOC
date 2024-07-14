# Supplementary code for the paper: Empirical estimators for risk-neutral composite optimal control with applications to bang-bang control

This repository contains supplementary code for the paper

> J. Milz and D. Walter: Empirical estimators for risk-neutral composite optimal control with applications to bang-bang control, working paper, Georgia Tech and HU Berlin, 2023.

## Abstract
Nonsmooth composite optimization problems under uncertainty are prevalent in various scientific and engineering applications.
We consider risk-neutral composite optimal control problems, where the objective function is the sum of
a potentially nonconvex expectation function and a nonsmooth convex function.
The expectation functions  are defined by solution operators of parameterized partial differential equations (PDEs).
To approximate the risk-neutral optimization problems, we use a  Monte Carlo sample-based approach,
study its asymptotic consistency, and derive nonasymptotic sample size estimates. 
Our analyses leverage problem structure commonly encountered in PDE-constrained optimization problems, including compact embeddings. We apply our findings to bang-bang-type optimal control problems and propose the use of a conditional gradient method to solve them effectively.
We present numerical illustrations.

## Getting started


### Docker

We provide a pre-build Docker image which can be used to run the the code in this repository. First thing you need to do is to ensure that you have [docker installed](https://docs.docker.com/get-docker/).

To start an interactive docker container you can execute the following command

```bash
docker run --rm -it ghcr.io/milzj/mcrnnoc:latest
```

### Running the simulations

### Postprocessing

### Installation without using Docker

```
conda env create -f environment.yml
conda activate MCRNNOC
```

## References

- The implementation of the [random field](./src/mcrnnoc/random_field) is adapted from
[poisson-risk-neutral-lognormal](https://github.com/milzj/FW4PDE/tree/main/examples/convex/poisson-risk-neutral-lognormal).
It implements the KKL expansion defined in Example 7.56 ("separable exponential $d=2$") with $a = a_1 = a_2 = 1/2$ in 
> G. J. Lord, C. E. Powell, and T. Shardlow. An Introduction to Computational Stochastic PDEs. Cambridge Texts Appl. Math. 50. Cambridge University Press, Cambridge, 2014. doi:10.1017/CBO9781139017329.

- We compute reference solutions using unscrambled Sobol' sequences. We shift the Sobol' sequences using a shift mentioned on p. 73 in
> A. B. Owen. On dropping the first Sobol’ point. In A. Keller, editor, Monte Carlo and quasi-Monte Carlo methods, Springer Proc. Math. Stat. 387, pages 71–86. Springer, Cham, 2022. doi:10.1007/978-3-030-98319-2\_4.

- We used [line_profiler and kernproof](https://github.com/pyutils/line_profiler).

## Having issues
If you have any troubles please file an issue in the GitHub repository.

## License
See [LICENSE](LICENSE)
