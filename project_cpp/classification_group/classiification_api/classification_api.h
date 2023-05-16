#pragma once
#ifdef CLASSIFICATIONAPI_EXPORTS
#define CLASSIFICATIONAPI_DLL_API __declspec( dllexport )
#else
#define CLASSIFICATIONAPI_DLL_API __declspec(dllimport)
#endif

extern "C" CLASSIFICATIONAPI_DLL_API void init_openvino( const char* s_model_xml, const char* s_model_bin, const char* s_device_name );
extern "C" CLASSIFICATIONAPI_DLL_API void exec_classification( void* img, int top_num, int* top_idx, float* top_Data );