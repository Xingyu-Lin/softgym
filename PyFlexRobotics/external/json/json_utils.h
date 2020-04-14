#ifndef NV_JSON_UTILS_H
#define NV_JSON_UTILS_H
#include "json.hpp"
#include <string>
#include <cstdlib>

template <class DataType>
inline DataType JsonGetOrExit(const nlohmann::json& jsonData, const std::string& key, const std::string& errorMessage = std::string())
{
	try
	{
		return jsonData.at(key);
	}
	catch (std::exception& e)
	{
		printf("%s\n", e.what());
		printf("Couldn't parse required key \"%s\"\n", key.c_str());
		if (!errorMessage.empty())
		{
			printf("%s\n", errorMessage.c_str());
		}
		exit(-1);
	}
}

#endif // !NV_JSON_UTILS_H