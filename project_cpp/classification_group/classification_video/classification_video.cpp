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

void init_openvino( std::string model_xml, std::string model_bin, std::string device_name ) {
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

void exec_classification(void* img, int top_num,int* top_idx, float* top_Data) {
	Blob::Ptr imgBlob = wrapMat2Blob( *((cv::Mat*)img ));
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

int main() {
	//std::string image_path = "E:\\user\\Documents\\OpenVINO\\demo_run\\classification_sample_async\\p.jpg";
	std::string model_xml = "Models\\googlenet-v3-pytorch.xml";
	std::string model_bin = "Models\\googlenet-v3-pytorch.bin";
	std::string labelFileName = "Models\\squeezenet1.1.labels";
	std::vector<std::string> labels;
	//cv::Mat image = cv::imread( "E:\\user\\Documents\\OpenVINO\\demo_run\\classification_sample_async\\p.jpg" );	

	init_openvino( model_xml, model_bin, "GPU" );

	std::ifstream inputFile;
	inputFile.open( labelFileName, std::ios::in );
	if ( inputFile.is_open() ) {
		std::string strLine;
		while ( std::getline( inputFile, strLine ) ) {
			trim( strLine );
			labels.push_back( strLine );
		}
	}
	int numClasses = labels.size();

	// 開啟 Webcam
	cv::VideoCapture cap( 0 );
	if ( !cap.isOpened() ) {
		std::cout << "無法開啟 Webcam" << std::endl;
		return -1;
	}

	//cap.set( cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, 1920 );
	//cap.set( cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, 1080 );

	cv::namedWindow( "Webcam", cv::WINDOW_KEEPRATIO );

	while ( cv::waitKey( 1 ) != 27 ) {
		cv::Mat frame;
		cap >> frame;

		int* top_idx = (int*)malloc( 5 * sizeof( int ) );
		float* top_data = (float*)malloc( 5 * sizeof( float ) );
		exec_classification( &frame, 5, top_idx, top_data );

		for ( size_t i = 0; i < 5; i++ ) {
			std::string label_name = labels[top_idx[i]];
			std::string score = std::to_string( top_data[i] );

			cv::putText( frame, label_name, cv::Point( 10, 30 + i * 80 ), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar( 50, 50, 200 ), 2 );
			cv::putText( frame, "Score: " + score, cv::Point( 10, 70 + i * 80 ), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar( 50, 50, 200 ), 2 );
		}

		// 顯示圖片
		cv::imshow( "Webcam", frame );
	}


}

/*
int main() {
	//std::string image_path = "E:\\user\\Documents\\OpenVINO\\demo_run\\classification_sample_async\\p.jpg";
	std::string model_xml = "C:\\Users\\willyhuang\\Documents\\Intel\\OpenVINO\\openvino_models\\ir\\public\\squeezenet1.1\\FP16\\squeezenet1.1.xml";
	std::string model_bin = "C:\\Users\\willyhuang\\Documents\\Intel\\OpenVINO\\openvino_models\\ir\\public\\squeezenet1.1\\FP16\\squeezenet1.1.bin";
	std::string labelFileName = "C:\\Users\\willyhuang\\Documents\\Intel\\OpenVINO\\openvino_models\\ir\\public\\squeezenet1.1\\FP16\\squeezenet1.1.labels";
	std::vector<std::string> labels;
	//cv::Mat image = cv::imread( "E:\\user\\Documents\\OpenVINO\\demo_run\\classification_sample_async\\p.jpg" );

	Core ie;
	CNNNetwork network = ie.ReadNetwork( model_xml, model_bin );	
	InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
	std::string input_name = network.getInputsInfo().begin()->first;

	input_info->getPreProcess().setResizeAlgorithm( RESIZE_BILINEAR );
	input_info->setLayout( Layout::NHWC );
	input_info->setPrecision( Precision::U8 );
	DataPtr output_info = network.getOutputsInfo().begin()->second;
	std::string output_name = network.getOutputsInfo().begin()->first;

	ExecutableNetwork executable_network = ie.LoadNetwork( network, "GPU" );
	InferRequest infer_request = executable_network.CreateInferRequest();	

	std::ifstream inputFile;
	inputFile.open( labelFileName, std::ios::in );
	if ( inputFile.is_open() ) {
		std::string strLine;
		while ( std::getline( inputFile, strLine ) ) {
			trim( strLine );
			labels.push_back( strLine );
		}
	}
	int numClasses = labels.size();

	// 開啟 Webcam
	cv::VideoCapture cap( 0 );
	if ( !cap.isOpened() ) {
		std::cout << "無法開啟 Webcam" << std::endl;
		return -1;
	}

	//cap.set( cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, 1920 );
	//cap.set( cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, 1080 );

	cv::namedWindow( "Webcam", cv::WINDOW_NORMAL );

	while ( cv::waitKey( 1 ) != 27 ) {
		cv::Mat frame;
		cap >> frame;

		Blob::Ptr imgBlob = wrapMat2Blob( frame );
		infer_request.SetBlob( input_name, imgBlob );
		infer_request.Infer();
		Blob::Ptr output = infer_request.GetBlob( output_name );
		// Print classification results
		//ClassificationResult classificationResult( output, { image_path } );
		//classificationResult.print();

		// 取得最大機率的分類索引
		float* outputData = output->buffer().as<float*>();
		printf( "element_size: %d\n", output->element_size() );
		printf( "size        : %d\n", output->size() );
		
		//int classIndex = std::max_element( outputData, outputData + numClasses ) - outputData;
		//float confidence = outputData[classIndex];

		// 取得分類標籤
		//std::string label = labels[classIndex];

		// 在圖片左上角繪製分類結果
		//cv::putText( frame, label, cv::Point( 10, 30 ), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar( 0, 0, 255 ), 2 );
		//cv::putText( frame, "Confidence: " + std::to_string( confidence ), cv::Point( 10, 70 ), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar( 0, 0, 255 ), 2 );
		
		// Get top 5 class indices and scores
		std::vector<std::pair<int, float>> classScores;
		for ( int i = 0; i < output->size(); i++ ) {
			classScores.push_back( { i, outputData[i] } );
		}

		sort( classScores.begin(), classScores.end(),
			[]( const std::pair<int, float>& a, const std::pair<int, float>& b ) {
				return a.second > b.second;
			} );
		classScores.resize( 5 );  // Keep only top 5

		int i = 0;
		// Draw class labels on the frame
		for ( const auto& classScore : classScores ) {
			// 取得分類標籤
			std::string label_name = labels[classScore.first];
			//std::string label = "Class: " + label_name + " Score: " + std::to_string( classScore.second );
			//cv::putText( frame, label, cv::Point( 10, 30 + ( ++i ) * 40 ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar( 0, 0, 255 ), 2 );
			cv::putText( frame, label_name, cv::Point( 10, 30 + ( i ) * 80 ), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar( 50, 50, 200 ), 2 );
			cv::putText( frame, "Score: " + std::to_string( classScore.second ), cv::Point( 10, 70 + ( i++ ) * 80 ), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar( 50, 50, 200 ), 2 );
		}
		// 顯示圖片
		cv::imshow( "Webcam", frame );
	}


}
*/