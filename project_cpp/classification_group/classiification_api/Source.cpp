#include "pch.h"
#include "classification_api.h"
#include <iostream>
#include <opencv2\opencv.hpp>
#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>
#include <samples/classification_results.h>

using namespace InferenceEngine;

Core ie;
CNNNetwork network;
std::string input_name;
std::string output_name;
InferRequest infer_request;

void init_openvino( const char* s_model_xml, const char* s_model_bin, const char* s_device_name ) {
	std::string model_xml( s_model_xml );
	std::string model_bin( s_model_bin );
	std::string device_name( s_device_name );
	network = ie.ReadNetwork( model_xml, model_bin );
	InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
	input_name = network.getInputsInfo().begin()->first;
	input_info->getPreProcess().setResizeAlgorithm( RESIZE_BILINEAR );
	input_info->setLayout( Layout::NHWC );
	input_info->setPrecision( Precision::U8 );
	DataPtr output_info = network.getOutputsInfo().begin()->second;
	output_name = network.getOutputsInfo().begin()->first;
	ExecutableNetwork executable_network = ie.LoadNetwork( network, device_name );
	infer_request = executable_network.CreateInferRequest();
}

void exec_classification( void* img, int top_num, int* top_idx, float* top_Data ) {
	Blob::Ptr imgBlob = wrapMat2Blob( *( (cv::Mat*)img ) );
	infer_request.SetBlob( input_name, imgBlob );
	infer_request.Infer();
	Blob::Ptr output = infer_request.GetBlob( output_name );

	// 取得最大機率的分類索引
	float* outputData = output->buffer().as<float*>();

	// Get top 5 class indices and scores
	std::vector<std::pair<int, float>> classScores;
	for ( int i = 0; i < output->size(); i++ ) {
		classScores.push_back( { i, outputData[i] } );
	}

	sort( classScores.begin(), classScores.end(),
		[]( const std::pair<int, float>& a, const std::pair<int, float>& b ) {
			return a.second > b.second;
		} );
	classScores.resize( top_num );  // Keep only top 5

	for ( size_t i = 0; i < top_num; i++ ) {
		top_idx[i] = classScores[i].first;
		top_Data[i] = classScores[i].second;
	}
}