#include "stdafx.h"
//#include "alg.h"


void Alg::build_n_RRsets(long long numRRsets)
{
	__pHyperG->build_n_RRsets(numRRsets);
	__numRRsets = __pHyperG->get_RRsets_size();
	__pRes->set_RRsets_size(__numRRsets);
}


double Alg::top_k_seed(int targetSize)
{
	Timer timerGreedy("greedy");
	double time1, time2;
	std::vector<std::pair<int, int>> coverageHeap(__numV);
	std::vector<int> coverage(__numV, 0);
	int largeDeg = 0;
	for (int i = 0; i < __numV; i++)
	{
		auto deg = (int)__pHyperG->_vecCover[i].size();
		coverage[i] = deg;
		coverageHeap[i] = std::pair<int, int>(std::make_pair(deg, i));
		if (deg > largeDeg) largeDeg = deg;
	}
	//auto largeDeg = *max_element(coverage.begin(), coverage.end());
	std::vector<std::vector<int>> vecCover(largeDeg + 1, std::vector<int>());
	for (int i = 0; i < __numV; i++)
	{
		if (coverage[i] == 0) continue;
		vecCover[coverage[i]].push_back(i);
	}
	time1 = timerGreedy.get_operation_time();
	int argmaxIdx;
	int sumInf = 0;

	// check if an edge is removed
	std::vector<bool> edgeMark(__numRRsets, false);

	__vecSeed.clear();
	for (auto deg = largeDeg + 1; deg--;)
	{
		auto& vecNode = vecCover[deg];
		for (auto idx = vecNode.size(); idx--;)
		{
			argmaxIdx = vecNode[idx];
			auto currDeg = coverage[argmaxIdx];
			if (deg > currDeg)
			{
				vecCover[currDeg].push_back(argmaxIdx);
				continue;
			}
			sumInf += currDeg;
			__vecSeed.push_back(argmaxIdx);
			if (__vecSeed.size() >= targetSize)
			{// Top-k influential nodes constructed
				time2 = timerGreedy.get_operation_time();
				auto time3 = timerGreedy.get_total_time();
				auto finalInfluence = 1.0 * sumInf * __numV / __numRRsets;
				std::cout << "\t==>greedy time: " << time1 << ", " << time2 << ", " << time3 << '\n';
				std::cout << "\t==>influence (origin): " << finalInfluence << ", verify: " << __pHyperG->eval_inf_spread(__vecSeed) << '\n';
				std::cout << "\t==>evaluate the solution quality...";
				auto realInf = __pHyperG->inf_valid_algo(__vecSeed, 100000);
				std::cout << "down!\n";
				std::cout << "\t==>influence (real): " << realInf << ", time used : " << timerGreedy.get_operation_time() << '\n';
				__pRes->set_running_time(time3);
				__pRes->set_influence(realInf);
				__pRes->set_influence_org(finalInfluence);
				__pRes->set_seed_vec(__vecSeed);
				return finalInfluence;
			}
			coverage[argmaxIdx] = 0;
			for (auto edgeIdx : __pHyperG->_vecCover[argmaxIdx]) {
				if (edgeMark[edgeIdx]) continue;
				edgeMark[edgeIdx] = true;
				for (auto nodeIdx : __pHyperG->_vecCoverRev[edgeIdx]) {
					coverage[nodeIdx]--;
				}
			}
		}
		vecCover.pop_back();
	}
	
	return 0.0;
}
