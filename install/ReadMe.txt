OpenVINO toolkit 2021.4.2 Install & Demo(Windows)

需先安裝Visual Studio 2017 or 2019

Step1. 安裝python (3.6~3.8)

*****請勿從Microsoft Store安裝Python*****

PYTHON下載網址: https://www.python.org/downloads/windows/

請下載包含x64位元.msi安裝包版本

安裝時請勾選加入系統變數Path

開啟cmd視窗：按下Win+R，輸入cmd，確定 

鍵入where python，會看到以下訊息
C:\Users\user\AppData\Local\Programs\Python\Python37\python.exe <- 安裝成功
C:\Users\user\AppData\Local\Microsoft\WindowsApps\python.exe <- 系統內建



Step2. 安裝cmake (3.13以上)

CMAKE下載網址: https://cmake.org/download/

請下載包含x64位元.msi安裝包版本

安裝時請勾選加入系統變數Path



Step3. 安裝w_openvino_toolkit_p_2021.4.752.exe

安裝時路徑不用修改

若跳出警告訊息(找不到python或cmake)，請按上一步並確認Step1, Step2是否正確安裝

確認完成後再按下一步，若沒出現警告即可繼續



Stop4. Run demo

請以系統管理員身分執行cmd：對開始按右鍵->搜尋->cmd->對命令提示字元按右鍵->以系統管理員身分執行

鍵入並執行 cd C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo
鍵入並執行 demo_squeezenet_download_convert_run.bat

若成功會印出分析結果

若失敗則進入Step5



Step5. 

進入以下路徑C:\Users\willyhuang\Documents\Intel\OpenVINO\openvino_models\ir\public\squeezenet1.1

刪除資料夾FP16

重新執行 demo_squeezenet_download_convert_run.bat

若失敗則須確認以下檔案是否存在
C:\Users\user\Documents\Intel\OpenVINO\openvino_models\ir\public\squeezenet1.1\FP16\squeezenet1.1.xml

若不存在則開啟cmd視窗

鍵入並執行 cd C:\Program Files (x86)\Intel\openvino_2021.4.752\deployment_tools\model_optimizer

model位置=C:\Users\willyhuang\Documents\Intel\OpenVINO\openvino_models\models\public\squeezenet1.1\squeezenet1.1.caffemodel
輸出位置=C:\Users\willyhuang\Documents\Intel\OpenVINO\openvino_models\ir\public\squeezenet1.1\FP16
鍵入並執行 mo_caffe.py --input_model "model位置" --output_dir "輸出位置"

確認有無產生squeezenet1.1.xml

若沒有則需確認錯誤訊息並排除


