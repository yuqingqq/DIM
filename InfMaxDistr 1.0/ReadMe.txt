This is a distributed verision of the IMM algorithm.

The settings are in argument.h, e.g., specify the dataset name, diffusion model (IC/LT), number of seeds k, etc.

The input of dataset (uncertain graph) is saved in the following format:
Graph[0]->List[pair<int,float>]: (neighbor1, probability1), (neighbor2, probability2)...
Graph[1]:...
...

It returns the estimated maximum influence of k seeds.

Run the execuatable file by:
	mpirun -n [number of processors] 'appname'
or mpiexec according to version of mpi.

