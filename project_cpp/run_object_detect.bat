cd "E:\Program Files (x86)\Intel\openvino_2021.4.752\deployment_tools\open_model_zoo\demos\intel64\Release"
object_detection_demo.exe -i "E:\user\Documents\OpenVINO\project_cpp\car-detection.mp4" -m "E:\user\Documents\OpenVINO\omz_model\public\yolo-v4-tiny-tf\FP16\yolo-v4-tiny-tf.xml" -at yolo -d GPU
cd E:\user\Documents\OpenVINO\project_cpp\