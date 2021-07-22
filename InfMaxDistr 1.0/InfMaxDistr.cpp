/**
* @file InfMaxDistr.cpp
* @brief This project is used to demonstrate the experiments for the influence maximization in distributed manner
*
* @author Jing Tang (Nanyang Technological University)
*
* Copyright (C) 2017 Jing Tang and Nanyang Technological University. All rights reserved.
*
*/

#include "stdafx.h"
#include <time.h>
#include <iomanip>
int targetSize = 50;
#define NATURAL_E 2.71828
//int buffer[NUM_PER_PROC];

//int* modified_all = (int*)calloc(NUM_PER_PROC * 50, sizeof(int));
//int modified_all[5*NUM_PER_PROC];
//int* rebuffer = (int*)calloc(NUM_PER_PROC * 50, sizeof(int));
//int rebuffer[10*NUM_PER_PROC];
//int* cover_all = (int*)calloc(NUM_PER_PROC * 50, sizeof(int));
//int cover_all[10*NUM_PER_PROC] = { 0 };
int max_index = 0;

int *vecSeed = (int *)calloc(targetSize, sizeof(int));
//int* modified_nodes = (int*)calloc(NUM_PER_PROC * 2, sizeof(int));
double commutime = 0;
double LB = 1;
double epsilon = 0.01, l = 1;
double epsilon_p = sqrt(2)*epsilon;
MPI_Request req;
MPI_Status status;
double solution_time = 0;
double total_time = 0;
double rrset_time = 0;
double all_time = 0;
double free_time = 0;
std::string outfilename = "face_lt.txt";
Timer t_free("");
inline double getGamma(double star, double el, double gamm, int num_nodes) {
	double left = 0, right = gamm;
	double result;
	while (1) {
		result = (left + right) / 2;
		if ((int)(star*(el + result)) + 1 <= pow(num_nodes, result))
			break;
		left = result;
	}
	return result;
}
inline double getFr(int world_size, int num_rr_sets, int num_nodes)
{
	Timer alltime("");
	int ithSeed = 0;
	int num_rr_set_each = num_rr_sets / (world_size - 1);
	for (int des = 1; des < world_size; des++) {
		MPI_Send(&num_rr_set_each, 1, MPI_INT, des, 0, MPI_COMM_WORLD);
	}
	Timer t_commu("");

	int max_nodes = 0, max_index = 0;
	double Fr = 0;
	//find index with maximum number of sets
	int num_degree;
	int * cover_all = (int*)calloc(num_nodes, sizeof(int));
	Timer t_rrset("");
	for (int source = 1; source < world_size; source++) {
		MPI_Recv(&num_degree, 1, MPI_INT, source, source, MPI_COMM_WORLD, &status);
		//if (num_nodes > max_num_nodes) {
		//	max_num_nodes = num_nodes;
		//}
		std::cout << "number of degrees from " << source << " is " << num_degree << std::endl;
		int * rebuffer = (int*)calloc(num_nodes, sizeof(int));
		//	t_commu.refresh_time();
		MPI_Irecv(rebuffer, num_nodes, MPI_INT, source, source, MPI_COMM_WORLD, &req);
		MPI_Wait(&req, &status);
		//	commutime += t_commu.get_operation_time();
		//std::cout << "communication time from " << source << " is " << commutime << std::endl;
		for (int i = 0; i < num_nodes; i++) {
			cover_all[i] += rebuffer[i];
		}
		free(rebuffer);
	}
	rrset_time += t_rrset.get_operation_time();
	//std::cout << "first round " << t.get_total_time() << std::endl;
	Timer solution("");

	solution.refresh_time();
	//t_1round.refresh_time();
	for (int i = 0; i < num_nodes; i++) {
		if (cover_all[i] > max_nodes) {
			max_nodes = cover_all[i];
			max_index = i;
		}
	}
	//first_round += t_1round.get_operation_time();
	//std::cout <<"1 round "<< first_round << std::endl;
	Fr += 1.0*cover_all[max_index] / num_rr_sets;
	//std::cout <<"parameters    !!!!!"<< cover_all[max_index]<<" "<<Fr <<" "<<num_rr_sets<< std::endl;
	std::vector<std::vector<int>> length_list(max_nodes + 1, std::vector<int>());
	for (int i = 0; i < num_nodes; i++) {
		length_list[cover_all[i]].push_back(i);
	}

	t_commu.refresh_time();
	for (int des = 1; des < world_size; des++) {
		MPI_Send(&max_index, 1, MPI_INT, des, 0, MPI_COMM_WORLD);
	}
	commutime += t_commu.get_operation_time();
	int num_modified_nodes;
	int arg = 0;

	vecSeed[ithSeed++] = max_index;
	//std::cout << max_index << std::endl;
	//std::cout << "number " << ithSeed << " iteration:index with maximum number of sets " << max_index << std::endl;
	for (int source = 1; source < world_size; source++) {

		MPI_Recv(&num_modified_nodes, 1, MPI_INT, source, source, MPI_COMM_WORLD, &status);
		int * modified_all = (int*)calloc(num_modified_nodes, sizeof(int));
		t_commu.refresh_time();
		MPI_Irecv(modified_all, num_modified_nodes, MPI_INT, source, source, MPI_COMM_WORLD, &req);
		MPI_Wait(&req, &status);
		commutime += t_commu.get_operation_time();
		for (int i = 0; i < num_modified_nodes; i++) {
			cover_all[modified_all[i]] -= modified_all[i + 1];
			i++;
		}
		t_free.refresh_time();
		free(modified_all);
		free_time += t_free.get_operation_time();
	}
	//cover_all[max_index] = 0;
	max_nodes = 0;
	max_index = 0;
	length_list.pop_back();
	//		while (vecSeed.size() <= targetSize){
	for (auto deg = length_list.size() - 1; deg >= 0; deg--) {
		auto& node = length_list[deg];
		for (auto idx = 0; idx<node.size(); idx++) {
			arg = node[idx];
			auto currDeg = cover_all[arg];

			if (deg > currDeg) {
				if (currDeg <= 0) {
					continue;
				}
				else {
					length_list[currDeg].push_back(arg);
					continue;
				}
			}
			Fr += 1.0* cover_all[arg] / num_rr_sets;
			//cover_all[arg] = 0;
			max_index = arg;
			vecSeed[ithSeed++] = max_index;
			if (ithSeed == targetSize) {
				t_free.refresh_time();
				free(cover_all);
				free_time += t_free.get_operation_time();
				solution_time += solution.get_operation_time();
				all_time += alltime.get_total_time();
				//std::cout << "all time = "<<all_time<< std::endl;
				//std::cout << "free time = "<<free_time << std::endl;
				//std::cout << "commu time = " << commutime << std::endl;
				//std::cout << "Fr = " << Fr << std::endl;
				return Fr;
			}
			t_commu.refresh_time();
			for (int des = 1; des < world_size; des++) {
				MPI_Send(&max_index, 1, MPI_INT, des, 0, MPI_COMM_WORLD);
			}
			commutime += t_commu.get_operation_time();
			//recv modified list / make changes

			for (int source = 1; source < world_size; source++) {

				MPI_Recv(&num_modified_nodes, 1, MPI_INT, source, source, MPI_COMM_WORLD, &status);
				int * modified_all = (int*)calloc(num_modified_nodes, sizeof(int));
				t_commu.refresh_time();
				MPI_Irecv(modified_all, num_modified_nodes, MPI_INT, source, source, MPI_COMM_WORLD, &req);
				MPI_Wait(&req, &status);
				commutime += t_commu.get_operation_time();
				for (int i = 0; i < num_modified_nodes; i += 2) {
					cover_all[modified_all[i]] -= modified_all[i + 1];
				}
				t_free.refresh_time();
				free(modified_all);
				free_time += t_free.get_operation_time();
			}
		}
		length_list.pop_back();
	}
	return 0;
}

int main(int argc, char* argv[])
{
	MPI_Init(NULL, NULL);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Request req;
	MPI_Status status;
	std::ofstream f1;
	
	if (rank == 0) {
		f1.open(outfilename, std::ios::app);
		std::cout << "here!!" << std::endl;
		if (!f1) return 0;
		f1 << std::endl;
		f1.close();
		double max_load_time = 0;
		double graph_load_time = 0;
		for (int source = 1; source < world_size; source++) {
			MPI_Recv(&graph_load_time, 1, MPI_DOUBLE, source, source, MPI_COMM_WORLD, &status);
			if (graph_load_time > max_load_time) {
				max_load_time = graph_load_time;
			}
			std::cout << "graph loaded time from " << source << " is " << graph_load_time << std::endl;
		}
		int num_nodes = 0, max_nodes = 0;
		double x, theta = 0;
		int num_rr_sets;
		double Fr = 0;
		MPI_Recv(&num_nodes, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &status);
		//l = l * (1 + log(2) / log(num_nodes));
		double alpha = sqrt(l*log(num_nodes) + log(2));
		double beta = sqrt((1 - 1 / NATURAL_E)*logcnk(num_nodes, targetSize) + l*log(num_nodes) + log(2));
		double lambda_p = (2 + 2 / 3 * epsilon_p)*(logcnk(num_nodes, targetSize) + l*log(num_nodes) + log(log2(num_nodes)))*num_nodes / pow(epsilon_p, 2);
		double lambda_s = 2 * num_nodes*pow2((1 - 1 / NATURAL_E)*alpha + beta)*pow2(1 / epsilon);
		std::cout << "lambda* was " << lambda_s << std::endl;
		std::cout << " lambda' was " << lambda_p << std::endl;
		double gamma = 4 + log(8 * log(num_nodes)) / log(num_nodes);
		gamma = getGamma(lambda_s, l, gamma, num_nodes);
		std::cout << "gamma = " << gamma << std::endl;
		l = l+ log(2) / log(num_nodes)+gamma;
		alpha = sqrt(l*log(num_nodes) + log(2));
		beta = sqrt((1 - 1 / NATURAL_E)*logcnk(num_nodes, targetSize) + l * log(num_nodes) + log(2));
		lambda_p = (2 + 2 / 3 * epsilon_p)*(logcnk(num_nodes, targetSize) + l * log(num_nodes) + log(log2(num_nodes)))*num_nodes / pow(epsilon_p, 2);
		lambda_s = 2 * num_nodes*pow2((1 - 1 / NATURAL_E)*alpha + beta)*pow2(1 / epsilon);
		std::cout << " lambda* = " << lambda_s << std::endl;
		std::cout << " lambda' = " << lambda_p << std::endl;
		Timer t_total("");
		for (int i = 1; i <= log2(num_nodes) - 1; i++)
		{
			x = num_nodes / pow(2, i);
			std::cout << "x = " << x << " lambda' = " << lambda_p << " number of RR sets " << int(lambda_p / x - theta) << std::endl;
			t_total.refresh_time();
			std::cout << "#rrsets = " << int(lambda_p / x - theta) << std::endl;
			Fr = getFr(world_size, int(lambda_p / x - theta), num_nodes);
			total_time += t_total.get_operation_time();
			//std::cout << i << " th round Fr = " <<Fr<< std::endl;
			//std::cout << "1+epsilon_p)*x = " << (1 + epsilon_p)*x << std::endl;
			theta = lambda_p / x;
			std::cout << "theta = " << theta << std::endl;
			if (num_nodes*Fr >= (1 + epsilon_p)*x)
			{
				LB = num_nodes*Fr / (1 + epsilon_p);
				break;
			}
		}
		double theta_p = lambda_s / LB;
		if (theta_p > theta)
		{
			t_total.refresh_time();
			Fr = getFr(world_size, int(theta_p - theta), num_nodes);
			std::cout << "final rr sets = " << int(theta_p - theta) << std::endl;
			total_time += t_total.get_operation_time();
		}
		//total_time += t_total.get_total_time();

		//terminate
		double t_rrset;
		num_rr_sets = 0;
		for (int des = 1; des < world_size; des++) {
			MPI_Send(&num_rr_sets, 1, MPI_INT, des, 0, MPI_COMM_WORLD);
		}
		f1.open(outfilename, std::ios::app);
		if (!f1) return 0;
		f1 << std::setw(20) << total_time << std::endl;
		f1 << std::setw(20) << commutime << std::endl;
		f1 << std::setw(20) << solution_time << std::endl;
		std::cout << "total time is " << total_time << std::endl;
		std::cout << "communication time " << commutime << std::endl;
		std::cout << "get solution time " << solution_time << std::endl;
		std::cout << " master rr set time " << rrset_time << std::endl;
		std::cout << "free time " << free_time << std::endl;

		for (int source = 1; source < world_size; source++) {
			MPI_Recv(&t_rrset, 1, MPI_DOUBLE, source, source, MPI_COMM_WORLD, &status);
			std::cout << "rrset time is " << t_rrset << std::endl;
			f1 << std::setw(20) << t_rrset << std::endl;
		}
		f1.close();
		MPI_Finalize();
		return 0;
	}

	//	}
	else {
		Timer t_worker("");
		// Randomize the seed for generating random numbers
		dsfmt_gv_init_gen_rand(static_cast<uint32_t>(time(nullptr)) + rank);
		TArgument Arg(argc, argv);
		std::string infilename = Arg._dir + "/" + Arg._graphname;
		if (Arg._func == 0 || Arg._func == 2)
		{// Format the graph
			GraphBase::format_graph(infilename, Arg._mode);
			if (Arg._func == 0) return 1;
		}
		std::cout << "---The Begin of " << Arg._outFileName << "---" << std::endl;
		//		Timer mainTimer("main");
		// Initialize a result object to record the results
		PResult pResult(new TResult());
		// Load the graph
		Graph graph = GraphBase::load_graph(infilename, true, Arg._probDist);
		if (Arg._model == TArgument::CascadeModel::LT)
		{// Normalize the propagation probabilities in accumulation format for LT cascade model for quickly generating RR sets
			to_normal_accum_prob(graph);
		}
		int numV = (int)graph.size();

		double lambda_p = (2 + 2 / 3 * epsilon_p)*(logcnk(numV, targetSize) + l*log(numV) + log(log2(numV)))*numV / pow(epsilon_p, 2);

		double* pCost;
		pCost = (double *)calloc(numV, sizeof(double));
		// Load the cost of each node
		//pCost = TIO::read_cost(infilename, numV, Arg._costDist, Arg._scale, Arg._para);
		//for (int i = 0; i < 20; i++) LogInfo(pCost[i]);
		double* pAccumWeight = nullptr; // Used for non-uniform benefit distribution for activating nodes
		bool isUniBen = true;
		if (tolower(Arg._benefitDist[0]) == 'n')
		{// Generate benefit weights for each node via normal distribution 
			isUniBen = false;
			/*Weighted benefit distribution*/
			std::default_random_engine generator;
			std::normal_distribution<double> distribution(3.0, 1.0);
			//std::binomial_distribution<int> distribution(100, 0.4);
			//std::poisson_distribution<int> distribution(1.0);
			//std::gamma_distribution<double> distribution(2.0, 2.0);
			//std::exponential_distribution<double> distribution(2.0);
			pAccumWeight = (double *)malloc(numV * sizeof(double));
			for (int i = 0; i < numV; i++)
			{
				pAccumWeight[i] = max(0, distribution(generator));
			}
			to_normal_accum_weight(pAccumWeight, numV);
		}
		// Create a hyper-graph object to generate/compute RR sets
		PHyperGraphRef pHyperG(new THyperGraphRef(graph, pAccumWeight));
		pHyperG->set_cascade_model(static_cast<THyperGraphRef::CascadeModel>(Arg._model));
		pHyperG->set_hyper_graph_mode(true);
		TAlg tAlg(pCost, pHyperG, pResult);
		std::cout << "  ==>Graph loaded for RIS! total time used (sec): " + std::to_string(t_worker.get_total_time()) << std::endl;
		//dsfmt_gv_init_gen_rand(0);
		double graph_load = t_worker.get_total_time();
		MPI_Send(&graph_load, 1, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
		if (rank == 1) {
			MPI_Send(&numV, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
		}
		//std::cout <<"num of V "<< numV << std::endl;
		int num_rr_sets;
		MPI_Recv(&num_rr_sets, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		// Build RR sets
		double t_RRset = 0;
		bool *modified_mark;
		modified_mark = (bool *)calloc(numV, sizeof(bool));

		int *modified_times;
		modified_times = (int *)calloc(numV, sizeof(int));

		int *modified_node_index;
		modified_node_index = (int *)calloc(numV, sizeof(int));
		while (1)
		{
			t_worker.refresh_time();
			tAlg.build_n_RRsets(num_rr_sets);
			t_RRset += t_worker.get_operation_time();
			std::cout << "from rank " << rank << " rr set time = " << t_RRset << std::endl;
			if (num_rr_sets == 0)
			{
				MPI_Send(&t_RRset, 1, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
				break;
			}
			int *coverage;
			coverage = (int *)calloc(numV, sizeof(int));
			int degrees = 0;
			for (int i = 0; i < numV; i++)
			{
				auto deg = (int)pHyperG->_vecCover[i].size();
				coverage[i] = deg;
				degrees += deg;
			}
			//send coverage[i]
			t_worker.refresh_time();
			MPI_Send(&degrees, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
			MPI_Isend(coverage, numV, MPI_INT, 0, rank, MPI_COMM_WORLD, &req);
			MPI_Wait(&req, &status);
			commutime += t_worker.get_operation_time();
			//receive max index 
			MPI_Recv(&max_index, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			//		printf("first iteration recved index %d at process %d\n", max_index, rank);
			//		time1 = timerGreedy.get_operation_time();
			int argmaxIdx, iteration = 1;
			bool * edgeMark;
			edgeMark = (bool *)calloc(pHyperG->get_RRsets_size(), sizeof(bool));

			while (iteration < targetSize) {
				int num_modi = 0;
				argmaxIdx = max_index;
				coverage[argmaxIdx] = 0;
				// check if an edge is removed	
				for (auto edgeIdx : pHyperG->_vecCover[argmaxIdx]) {
					if (edgeMark[edgeIdx]) continue;
					edgeMark[edgeIdx] = true;
					for (auto nodeIdx : pHyperG->_vecCoverRev[edgeIdx]) {
						if (modified_mark[nodeIdx] != true)
						{
							modified_mark[nodeIdx] = true;
							modified_node_index[num_modi++] = nodeIdx;
						}
						modified_times[nodeIdx]++;
						coverage[nodeIdx]--;
						//					modified_nodes[num_modi] = nodeIdx;
						//					num_modi++;
					}
				}
				t_free.refresh_time();
				int * modified_nodes = (int *)calloc(num_modi * 2, sizeof(int));
				for (auto idx = 0; idx < num_modi; idx++)
				{
					auto nodeIdx = modified_node_index[idx];
					modified_nodes[2 * idx] = nodeIdx;
					modified_nodes[2 * idx + 1] = modified_times[nodeIdx];
					modified_mark[nodeIdx] = false;
					modified_times[nodeIdx] = 0;
				}
				free_time += t_free.get_operation_time();
				//	
				//	MPI_Send(&num_modi, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
				//	MPI_Isend(modified_nodes, num_modi, MPI_INT, 0, rank, MPI_COMM_WORLD, &req);
				t_worker.refresh_time();
				MPI_Send(&num_modi, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
				MPI_Isend(modified_nodes, num_modi, MPI_INT, 0, rank, MPI_COMM_WORLD, &req);
				MPI_Wait(&req, &status);
				commutime += t_worker.get_operation_time();
				if (iteration == targetSize - 1) {
					break;
				}
				t_worker.refresh_time();
				MPI_Recv(&max_index, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
				commutime += t_worker.get_operation_time();
				//			printf("%d iteration recved index %d at process %d\n", iteration + 1, max_index, rank);
				iteration++;

				t_free.refresh_time();
				free(modified_nodes);
				free_time += t_free.get_operation_time();
			}
			t_free.refresh_time();
			free(edgeMark);
			free(coverage);
			free_time += t_free.get_operation_time();
			t_worker.refresh_time();
			MPI_Recv(&num_rr_sets, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			commutime += t_worker.get_operation_time();

			//std::cout << "finish time of rank " << rank << " is " << std::to_string(t_worker.get_total_time()) << std::endl;
		}
		t_free.refresh_time();
		free(modified_mark);
		free(modified_times);
		free(modified_node_index);
		free_time += t_free.get_operation_time();
	}
	MPI_Finalize();
	return 0;
}