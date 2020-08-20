# yolov4

Weights of pre-trained model
https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo
COCO pre-trained files (416x416)
Models used anchors of 416x416 input resolution for training.
Model	Size	AP	AP50	AP75	APS	APM	APL	cfg	weights	score
YOLOv4-Leaky-416	416	40.7	62.7	43.9	21.4	43.7	54.0	cfg	weights	score
YOLOv4-Mish-416	416	41.5	63.3	44.7	21.9	44.4	55.3	cfg	weights	score

From <https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo> 


# in ubuntu 18.04 container
apt update && apt upgrade -y
apt install -y git vim pciutils python3-dev python3-setuptools python3-wheel
apt install -y python3-pip python3-venv

python3 -m venv tf1.14_venv
source tf1.14_venv/bin/activate
pip3 install setuptools wheel
pip3 install tensorflow==1.14
pip3 install pillow gdown
pip3 install gast==0.2.2

git clone https://github.com/TNTWEN/OpenVINO-YOLOV4
cd OpenVINO-YOLOV4/

#download YOLOv4-Mish-416 weight from
https://drive.google.com/open?id=1NuYL-MBKU0ko0dwsvnCx6vUr7970XSAR

gdown --id 1NuYL-MBKU0ko0dwsvnCx6vUr7970XSAR
python convert_weights_pb.py --class_names cfg\coco.names --weight_file yolov4-mish-416.weights --data_format NHWC

cd yolov4-relu/

#download YOLOv4-Leaky-416 weight from
https://drive.google.com/open?id=1bV4RyU_-PNB78G-OtoTmw1Q7t_q90GKY 
gdown --id 1bV4RyU_-PNB78G-OtoTmw1Q7t_q90GKY

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-leaky-416.weights --data_format NHWC

wget https://apt.repos.intel.com/openvino/2020/GPG-PUB-KEY-INTEL-OPENVINO-2020
apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2020
touch /etc/apt/sources.list.d/intel-openvino-2020.list
echo "deb https://apt.repos.intel.com/openvino/2020 all main" >> /etc/apt/sources.list.d/intel-openvino-2020.list
apt update
apt install intel-openvino-dev-ubuntu18-2020.4.287

##apt install sudo
##/opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites_tf.sh

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_darknet_yolov4_model.pb --transformations_config yolov4.json --batch 1 --reverse_input_channels --model frozen_darknet_yolov4-mish-416_FP16_model --data_type FP16

python3 /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py -m frozen_darknet_yolov4-mish-416_model.xml

python3 /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py -m yolov4-relu/frozen_darknet_yolov4-leaky-416
