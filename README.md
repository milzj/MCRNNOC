# MCRNNOC

The code developed here is used in the working paper

> J. Milz and D. Walter: Monte Carlo estimators for risk-neutral, nonsmooth optimal control with applications to bang-bang problems, working paper, Georgia Tech and HU Berlin, 2023.

## Installation

```
conda env create -f environment.yml
conda activate MCRNNOC
```


## Examples

## References

- The implementation of the [random field](./mcrnnoc/random_field) is adapted from
[poisson-risk-neutral-lognormal](https://github.com/milzj/FW4PDE/tree/main/examples/convex/poisson-risk-neutral-lognormal).
The implementation is adapted in that implementation details are added. 
It implements the KL expansion defined in Example 7.56 ("separable exponential $d=2$") with $a = a_1 = a_2 = 1/2$ in 
> G. J. Lord, C. E. Powell, and T. Shardlow. An Introduction to Computational Stochastic PDEs. Cambridge Texts Appl. Math. 50. Cambridge University Press, Cambridge, 2014. doi:10.1017/CBO9781139017329.

- We compute reference solutions using unscrambled Sobol' sequences. We shift the Sobol' sequences using a shift mentioned on p. 73 in
> A. B. Owen. On dropping the first Sobol’ point. In A. Keller, editor, Monte Carlo and quasi-Monte Carlo methods, Springer Proc. Math. Stat. 387, pages 71–86. Springer, Cham, 2022. doi:10.1007/978-3-030-98319-2\_4.

- We used [line_profiler and kernproof](https://github.com/pyutils/line_profiler).

## Authors

- [Johannes Milz](https://www.isye.gatech.edu/users/johannes-milz)
- [Daniel Walter](https://www.mathematik.hu-berlin.de/de/personen/mitarb-vz/1694929)
