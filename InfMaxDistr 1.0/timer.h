#pragma once

#ifdef _WIN32 
#include <windows.h> 
class Timer
{
private:
	LARGE_INTEGER __StartTime, __LastTime, __EndTime, __TimeFreq;
	const char* __processName;
public:
	Timer()
	{
		QueryPerformanceFrequency(&__TimeFreq);
		QueryPerformanceCounter(&__StartTime);
		__LastTime = __StartTime, __EndTime = __StartTime;
		__processName = "Unnamed";
	}

	explicit Timer(const char* processName)
	{
		QueryPerformanceFrequency(&__TimeFreq);
		QueryPerformanceCounter(&__StartTime);
		__LastTime = __StartTime, __EndTime = __StartTime;
		__processName = processName;
	}

	/// Refresh time
	void refresh_time()
	{
		QueryPerformanceCounter(&__StartTime);
		__LastTime = __StartTime, __EndTime = __StartTime;
	}

	/// Record current time
	void record_current_time()
	{
		__LastTime = __EndTime;
		QueryPerformanceCounter(&__EndTime);
	};
	/// Get the operation time from the last recorded moment
	double get_operation_time()
	{
		this->record_current_time();
		return (double)(__EndTime.QuadPart - __LastTime.QuadPart) / __TimeFreq.QuadPart;
	}

	/// Print the operation time from the last recorded moment
	void log_operation_time()
	{
		double duration = this->get_operation_time();
		std::cout << "->Time used (sec): " << duration << std::endl;
	}

	/// Print the operation time from the last recorded moment with a given name
	void log_operation_time(const char* operationName)
	{
		double duration = this->get_operation_time();
		std::cout << "->Time used (sec) for operation [" << operationName << "]: " << duration << std::endl;
	}

	/// Get the total time from the beginning
	double get_total_time()
	{
		this->record_current_time();
		return (double)(__EndTime.QuadPart - __StartTime.QuadPart) / __TimeFreq.QuadPart;
	}

	/// Print the total time from the beginning
	void log_total_time()
	{
		double duration = this->get_total_time();
		std::cout << "--->Time used (sec) for process [" << __processName << "]: " << duration << std::endl;
	}

	/// Print the total time from the beginning to the last recorded moment
	void log_sub_total_time() const
	{
		double duration = (double)(__EndTime.QuadPart - __StartTime.QuadPart) / __TimeFreq.QuadPart;
		std::cout << "--->Time used (sec) for process [" << __processName << "]: " << duration << std::endl;
	}
};
#else 
#define _BSD_SOURCE
#include <sys/time.h>
class Timer
{
private:
	struct timeval __StartTime, __LastTime, __EndTime;
	const char* __processName;
public:
	Timer()
	{
		gettimeofday(&__StartTime, NULL);
		gettimeofday(&__LastTime, NULL);
		__processName = "Unnamed";
	}

	explicit Timer(const char* processName)
	{
		gettimeofday(&__StartTime, NULL);
		gettimeofday(&__LastTime, NULL);
		__processName = processName;
	}

	/// Refresh time
	void refresh_time()
	{
		gettimeofday(&__LastTime, NULL);
	}

	/// Record current time
	void record_current_time()
	{
		gettimeofday(&__LastTime, NULL);
	};
	/// Get the operation time from the last recorded moment
	double get_operation_time()
	{
		gettimeofday(&__EndTime, NULL);
		return (__EndTime.tv_sec - __LastTime.tv_sec) + (__EndTime.tv_usec - __LastTime.tv_usec) / 1000000.0;
	}

	/// Print the operation time from the last recorded moment
	void log_operation_time()
	{
		gettimeofday(&__EndTime, NULL);
		double duration = (__EndTime.tv_sec - __LastTime.tv_sec) + (__EndTime.tv_usec - __LastTime.tv_usec) / 1000000.0;
		std::cout << "->Time used (sec): " << duration << std::endl;
	}

	/// Print the operation time from the last recorded moment with a given name
	void log_operation_time(const char* operationName)
	{
		gettimeofday(&__EndTime, NULL);
		double duration = (__EndTime.tv_sec - __LastTime.tv_sec)+ (__EndTime.tv_usec - __LastTime.tv_usec) / 1000000.0;
		std::cout << "->Time used (sec) for operation [" << operationName << "]: " << duration << std::endl;
	}

	/// Get the total time from the beginning
	double get_total_time()
	{
		gettimeofday(&__EndTime, NULL);
		return  (__EndTime.tv_sec - __StartTime.tv_sec) + (__EndTime.tv_usec - __StartTime.tv_usec) / 1000000.0;
	}

	/// Print the total time from the beginning
	void log_total_time()
	{
		gettimeofday(&__EndTime, NULL);
		double duration = (__EndTime.tv_sec - __LastTime.tv_sec) + (__EndTime.tv_usec - __LastTime.tv_usec) / 1000000.0;
		std::cout << "--->Time used (sec) for process [" << __processName << "]: " << duration << std::endl;
	}

	/// Print the total time from the beginning to the last recorded moment
	void log_sub_total_time()
	{	
		gettimeofday(&__EndTime, NULL);
		double duration = (__EndTime.tv_sec - __LastTime.tv_sec) + (__EndTime.tv_usec - __LastTime.tv_usec) / 1000000.0;
		std::cout << "--->Time used (sec) for process [" << __processName << "]: " << duration << std::endl;
	}
};
//#include <sys/time.h> 
//class Timer {
//public: 
//	void Start()
//{
//	gettimeofday(&m_start, NULL);
//}
//		void End()
//		{
//			gettimeofday(&m_end, NULL);
//		}
//		void GetDelta()
//		{
//			return (m_end.tv_sec - m_start.tv_sec) * 1000.0 + (m_end.tv_usec - m_start.tv_usec) / 1000.0;
//		}
//private: struct timeval m_start;
//		 struct timeval m_end;
//};
#endif 
using TTimer = Timer;
using PTimer = std::shared_ptr<Timer>;
