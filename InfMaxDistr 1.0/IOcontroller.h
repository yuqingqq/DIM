#pragma once

class IOcontroller
{
public:
	/// Save a serialized file
	template <class T>
	static void save_file(std::string filename, const T& output)
	{
		std::ofstream outfile(filename, std::ios::binary);
		if (!outfile.eof() && !outfile.fail())
		{
			StreamType res;
			serialize(output, res);
			outfile.write(reinterpret_cast<char*>(&res[0]), res.size());
			outfile.close();
			res.clear();
			std::cout << "Save file successfully: " << filename << '\n';
		}
		else
		{
			std::cout << "Save file failed: " + filename << '\n';
			exit(1);
		}
	}

	/// Load a serialized file
	template <class T>
	static void load_file(std::string filename, T& input)
	{
		//Timer t("");
		std::ifstream infile(filename, std::ios::binary);
		if (!infile.eof() && !infile.fail())
		{
			infile.seekg(0, std::ios_base::end);
			std::streampos fileSize = infile.tellg();
			infile.seekg(0, std::ios_base::beg);
		//	t.log_operation_time("file read time ");
			std::vector<uint8_t> res(fileSize);
			infile.read(reinterpret_cast<char*>(&res[0]), fileSize);
		//	t.log_operation_time("file read time ");
			infile.close();
			input.clear();
			auto it = res.cbegin();
			input = deserialize<T>(it, res.cend());
			res.clear();
		//	t.log_operation_time("file load ");
		}
		else
		{
			std::cout << "Cannot open file: " + filename << '\n';
			exit(1);
		}
	}

	/// Save graph structure to a file
	static void save_graph_struct(std::string graphName, const Graph& vecGraph, const bool isReverse)
	{
		std::string postfix = ".vec.graph.property";
		if (isReverse) postfix = ".vec.rvs.graph.property";
		std::string filename = graphName + postfix;
		save_file(filename, vecGraph);
	}

	/// Load graph structure from a file
	static void load_graph_struct(std::string graphName, Graph& vecGraph, const bool isReverse)
	{
		std::string postfix = ".vec.graph.property";
		if (isReverse) postfix = ".vec.rvs.graph.property";
		std::string filename = graphName + postfix;
		load_file(filename, vecGraph);
	}

	/// Load cost of each node from a file
	static double* read_cost(std::string graphName, int numV, std::string costType, double scale, double base)
	{
		std::string fullName = graphName + "." + costType + ".cost";
		double *cost, val;
		cost = (double *)malloc(sizeof(double) * numV);
		memset(cost, 0, sizeof(double) * numV);
		std::ifstream costfile(fullName);
		base = scale * base;
		scale -= base;
		if (costfile.is_open())
		{
			for (int i = 0; i < numV; i++)
			{
				costfile >> val;
				cost[i] = scale * val + base;
			}
			costfile.close();
		}
		else
		{
			std::cout << "Cannot open file: " + fullName << '\n';
			exit(0);
		}
		return cost;
	}

	/// Get out-file name
	static std::string get_out_file_name(std::string graphName, std::string algName, int size)
	{
		std::ostringstream strs1;
		strs1 << size;
		std::string strSize = strs1.str();
		return graphName + "_" + algName + "_k" + strSize;
	}

	/// Print the results
	static void write_result(std::string outFileName, PResult resultObj, std::string outFolder)
	{
		double runTime = resultObj->get_running_time();
		double influence = resultObj->get_influence();
		double influenceOrg = resultObj->get_influence_org();
		int seedSize = resultObj->get_seed_size();
		int RRsetsSize = resultObj->get_RRsets_size();
		std::cout << "   --------------------" << std::endl;
		std::cout << "  |Time (sec): " << runTime << std::endl;
		std::cout << "  |influence: " << influence << std::endl;
		std::cout << "  |influence (origin): " << influenceOrg << std::endl;
		std::cout << "  |#Seeds: " << seedSize << std::endl;
		std::cout << "  |#RR sets: " << RRsetsSize << std::endl;
		std::cout << "   --------------------" << std::endl;
		/*CreateDirectoryA(outFolder.c_str(), nullptr);
		std::ofstream outFileNew(outFolder + "/" + "performance_" + outFileName);
		if (outFileNew.is_open())
		{
			outFileNew << "time (sec): " << runTime << '\n';
			outFileNew << "influence: " << influence << '\n';
			outFileNew << "influence (origin): " << influenceOrg << '\n';
			outFileNew << "seed size: " << seedSize << '\n';
			outFileNew << "#RR sets: " << RRsetsSize << '\n';
			outFileNew.close();
		}*/
	}

	/// Print the seeds
	static void write_order_seeds(std::string outFileName, PResult resultObj, std::string outFolder)
	{
		/*CreateDirectoryA(outFolder.c_str(), nullptr);
		auto vecSeed = resultObj->get_seed_vec();
		std::ofstream outFile(outFolder + "/seed_" + outFileName);
		for (auto i = 0; i < vecSeed.size(); i++)
		{
			outFile << vecSeed[i] << '\n';
		}
		outFile.close();*/
	}
};

using TIO = IOcontroller;
using PIO = std::shared_ptr<IOcontroller>;
