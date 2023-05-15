// object_detect_video.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//
#include <iostream>
#include <opencv2\opencv.hpp>
#include <inference_engine.hpp>
#include <samples\ocv_common.hpp>

using namespace InferenceEngine;

int main() {
    try {
        // 載入模型和設定
        std::string modelPath = "E:\\user\\Documents\\OpenVINO\\omz_model\\public\\yolo-v4-tiny-tf\\FP32\\yolo-v4-tiny-tf.xml";
        std::string weightsPath = "E:\\user\\Documents\\OpenVINO\\omz_model\\public\\yolo-v4-tiny-tf\\FP32\\yolo-v4-tiny-tf.bin";
        std::string device = "GPU";  // 可以改為"GPU"或"MYRIAD"等

        // 載入網路
        Core ie;
        CNNNetwork network = ie.ReadNetwork( modelPath, weightsPath );
        network.setBatchSize( 1 );

        // 設定輸入和輸出
        InputsDataMap inputInfo( network.getInputsInfo() );
        InputInfo::Ptr& input = inputInfo.begin()->second;
        std::string input_name = network.getInputsInfo().begin()->first;
        input->setPrecision( Precision::U8 );
        input->setLayout( Layout::NCHW );
        input->getInputData()->setLayout( Layout::NCHW );
        input->getPreProcess().setResizeAlgorithm( ResizeAlgorithm::RESIZE_BILINEAR );
        //input->getPreProcess().setColorFormat( ColorFormat::RGB );

        OutputsDataMap outputInfo( network.getOutputsInfo() );
        DataPtr& output = outputInfo.begin()->second;
        output->setPrecision( Precision::FP32 );
        std::string output_name = outputInfo.begin()->first;

        // 載入網路至裝置
        ExecutableNetwork executableNetwork = ie.LoadNetwork( network, device );

        // 創建推論輸出
        InferRequest inferRequest = executableNetwork.CreateInferRequest();

        // 開啟Webcam
        cv::VideoCapture cap( 0 );
        if ( !cap.isOpened() ) {
            std::cerr << "無法開啟Webcam!" << std::endl;
            return 1;
        }

        cv::Mat frame;
        while ( true ) {
            // 擷取Webcam影像
            cap >> frame;
            if ( frame.empty() )
                break;

            Blob::Ptr imgBlob = wrapMat2Blob( frame );
            inferRequest.SetBlob( input_name, imgBlob );

            // 執行推論
            inferRequest.Infer();

            // 取得輸出結果
            Blob::Ptr outputBlob = inferRequest.GetBlob( output_name );
            const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>( outputBlob->buffer() );

            // 解析輸出結果
            const float* detectionResults = detection;
            for ( int i = 0; i < 100; i++ ) {
                float image_id = detectionResults[i * 7 + 0];
                if ( image_id < 0 )
                    break;

                float label = detectionResults[i * 7 + 1];
                float confidence = detectionResults[i * 7 + 2];
                float x_min = detectionResults[i * 7 + 3] * frame.cols;
                float y_min = detectionResults[i * 7 + 4] * frame.rows;
                float x_max = detectionResults[i * 7 + 5] * frame.cols;
                float y_max = detectionResults[i * 7 + 6] * frame.rows;

                if ( confidence > 0.5 ) {
                    cv::rectangle( frame, cv::Point( x_min, y_min ), cv::Point( x_max, y_max ), cv::Scalar( 0, 255, 0 ), 2 );
                    cv::putText( frame, std::to_string( label ), cv::Point( x_min, y_min - 5 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar( 0, 255, 0 ), 2 );
                }
            }

            // 顯示結果
            cv::imshow( "Object Detection", frame );
            if ( cv::waitKey( 1 ) == 27 )  // 按下ESC鍵退出
                break;
        }

        // 釋放資源
        cap.release();
        cv::destroyAllWindows();
    }
    catch ( const std::exception& error ) {
        std::cerr << "發生錯誤: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
