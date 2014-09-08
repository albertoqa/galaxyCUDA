galaxyCUDA
==========

In this project I wanted to prove the efficiency of GPUs over CPUs for scientific
calculations. 

*Problem: calculate the two-point angular correlation function for two sets of galaxies.*

* Implementations:
  * C implementation, sequential (no parallel programming)
  * CUDA implementation, parallel programming
  
I've obtained a code speed-up of 70x faster (on average) compared to computing the same calculation on the CPU.


***More details on the info file***


* There are some auxiliar files attached to this project:
  * hello world, CUDA programming
  * simple sum, CUDA programming