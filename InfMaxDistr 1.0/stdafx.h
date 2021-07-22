// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once
#include <limits.h>
#include <climits>
#include "mpi.h"
#include "targetver.h"

#include <stdio.h>
//#include <tchar.h>
#if defined(__BORLANDC__) && !defined(_TCHAR_DEFINED)
	typedef _TCHAR    TCHAR, *PTCHAR;
	typedef _TCHAR    TBYTE, *PTBYTE;
	#define _TCHAR_DEFINED
#endif    // #if defined(__BORLANDC__) && !defined(_TCHAR_DEFINED)
// TODO: reference additional headers your program requires here
#if defined(_MSC_VER) || defined(__BORLANDC__)
#if !defined(DSFMT_UINT32_DEFINED) && !defined(SFMT_UINT32_DEFINED)
typedef __int64 int64;
#endif
#endif
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <ctime>
#include <memory>
#include <random>
#include <functional>
#include <math.h>
#include <iomanip>
#include "dSFMT.h"
#include "timer.h"
#include "commonStruct.h"
#include "commonFunc.h"
#include "heap.h"
#include "serialize.h"
#include "resultInfo.h"
#include "IOcontroller.h"
#include "argument.h"
#include "graphBase.h"
#include "hyperGraphRef.h"
#include <cmath>
#include "alg.h"
