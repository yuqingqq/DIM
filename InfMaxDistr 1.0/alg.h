#pragma once

class Alg
{
private:
	int __numV, __numRRsets;
	std::vector<int> __vecSeed;
	const double* __pCost;
	PHyperGraphRef __pHyperG;
	PResult __pRes;

public:
	Alg(const double* pCost, const PHyperGraphRef& pHyperG, const PResult& pRes) : __pCost(pCost), __pHyperG(pHyperG), __pRes(pRes)
	{
		__numV = __pHyperG->get_nodes();
		__numRRsets = __pHyperG->get_RRsets_size();
	}

	/// Build a set of n RR sets
	void build_n_RRsets(long long numRRsets);
	/// Find the top-k seeds
	double top_k_seed(int targetSize);
};

using TAlg = Alg;
using PAlg = std::shared_ptr<TAlg>;

