#pragma once

class ResultInfo
{
private:
	double __RunningTime, __Influence, __InfluenceOrg;
	int __SeedSize, __EstNodeSize, __RRsetsSize;
	std::vector<int> __VecSeed;
public:
	ResultInfo()
	{
		reflesh();
	}

	~ResultInfo()
	{
	}

	void reflesh()
	{
		__RunningTime = -1.0;
		__Influence = -1.0;
		__InfluenceOrg = -1.0;
		__SeedSize = 0;
		__EstNodeSize = 0;
		__RRsetsSize = -1;
		__VecSeed.clear();
	}

	double get_running_time() const
	{
		return __RunningTime;
	}

	double get_influence() const
	{
		return __Influence;
	}

	double get_influence_org() const
	{
		return __InfluenceOrg;
	}

	int get_seed_size() const
	{
		return __SeedSize;
	}

	int get_estimated_node_size() const
	{
		return __EstNodeSize;
	}

	int get_RRsets_size() const
	{
		return __RRsetsSize;
	}

	std::vector<int> get_seed_vec() const
	{
		return __VecSeed;
	}

	void set_running_time(double value)
	{
		__RunningTime = value;
	}

	void set_influence(double value)
	{
		__Influence = value;
	}

	void set_influence_org(double value)
	{
		__InfluenceOrg = value;
	}

	void set_seed_size(int value)
	{
		__SeedSize = value;
	}

	void set_estimated_node_size(int value)
	{
		__EstNodeSize = value;
	}

	void set_RRsets_size(int value)
	{
		__RRsetsSize = value;
	}

	void inc_estimated_node_size(int value)
	{
		__EstNodeSize += value;
	}

	void set_seed_vec(std::vector<int>& vecSeed)
	{
		__VecSeed = vecSeed;
		set_seed_size((int)vecSeed.size());
	}
};

using TResult = ResultInfo;
using PResult = std::shared_ptr<ResultInfo>;
