#include <iostream>
#include <opencv2\opencv.hpp>
#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>
#include <samples/classification_results.h>

using namespace InferenceEngine;

int main() {
	std::string image_path = "E:\\user\\Documents\\OpenVINO\\demo_run\\classification_sample_async\\p.jpg";
	std::string model_xml = "E:\\user\\Documents\\OpenVINO\\demo_run\\classification_sample_async\\model\\squeezenet1.1.xml";
	std::string model_bin = "E:\\user\\Documents\\OpenVINO\\demo_run\\classification_sample_async\\model\\squeezenet1.1.bin";
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

	std::string labelFileName = "E:\\user\\Documents\\OpenVINO\\demo_run\\classification_sample_async\\model\\squeezenet1.1.labels";
	std::vector<std::string> labels;

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
		int classIndex = std::max_element( outputData, outputData + numClasses ) - outputData;
		float confidence = outputData[classIndex];

		// 取得分類標籤
		std::string label = labels[classIndex];

		// 在圖片左上角繪製分類結果
		cv::putText( frame, label, cv::Point( 10, 30 ), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar( 0, 0, 255 ), 2 );
		cv::putText( frame, "Confidence: " + std::to_string( confidence ), cv::Point( 10, 70 ), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar( 0, 0, 255 ), 2 );

		// 顯示圖片
		cv::imshow( "Webcam", frame );
	}


}