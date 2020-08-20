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

pip3 install pillow gdown

# https://github.com/tensorflow/tensorflow/issues/32383
pip3 install gast==0.2.2

pip3 install tensorflow==1.14

git clone https://github.com/TNTWEN/OpenVINO-YOLOV4
cd OpenVINO-YOLOV4/

#download YOLOv4-Mish-416 weight from
https://drive.google.com/open?id=1NuYL-MBKU0ko0dwsvnCx6vUr7970XSAR

gdown --id 1NuYL-MBKU0ko0dwsvnCx6vUr7970XSAR

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-mish-416.weights --data_format NHWC

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

pip3 install -r /opt/intel/openvino/deployment_tools/model_optimizer/requirements_tf.txt

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_darknet_yolov4_model.pb --transformations_config yolov4.json --batch 1 --reverse_input_channels --model frozen_darknet_yolov4-mish-416_FP16_model --data_type FP16

cd yolov4-relu

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_darknet_yolov4_model.pb --transformations_config yolov4.json --batch 1 --reverse_input_channels --model frozen_darknet_yolov4-leaky-416_FP16_model --data_type FP16

pip3 install progress

source /opt/intel/openvino/bin/setupvars.sh

python3 /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py -m frozen_darknet_yolov4-mish-416_FP16_model.xml

(tf1.14_venv) dlcv@dlcv-NUC7i7BNH:~/OpenVINO-YOLOV4$ python3 /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py -m yolov4-relu/frozen_darknet_yolov4-leaky-416_FP16_model.xml
[Step 1/11] Parsing and validating input arguments
/opt/intel/openvino_2020.4.287/python/python3.6/openvino/tools/benchmark/main.py:29: DeprecationWarning: The 'warn' method is deprecated, use 'warning' instead
  logger.warn(" -nstreams default value is determined automatically for a device. "
[ WARNING ]  -nstreams default value is determined automatically for a device. Although the automatic selection usually provides a reasonable performance, but it still may be non-optimal for some cases, for more information look at README. 
[Step 2/11] Loading Inference Engine
[ INFO ] InferenceEngine:
         API version............. 2.1.2020.4.0-359-21e092122f4-releases/2020/4
[ INFO ] Device info
         CPU
         MKLDNNPlugin............ version 2.1
         Build................... 2020.4.0-359-21e092122f4-releases/2020/4

[Step 3/11] Setting device configuration
[ WARNING ] -nstreams default value is determined automatically for CPU device. Although the automatic selection usually provides a reasonable performance,but it still may be non-optimal for some cases, for more information look at README.
[Step 4/11] Reading the Intermediate Representation network
[ INFO ] Read network took 113.06 ms
[Step 5/11] Resizing network to match image sizes and given batch
[ INFO ] Network batch size: 1
[Step 6/11] Configuring input of the model
[Step 7/11] Loading the model to the device
[ INFO ] Load network took 1993.37 ms
[Step 8/11] Setting optimal runtime parameters
[Step 9/11] Creating infer requests and filling input blobs with images
[ INFO ] Network input 'inputs' precision U8, dimensions (NCHW): 1 3 416 416
/opt/intel/openvino_2020.4.287/python/python3.6/openvino/tools/benchmark/utils/inputs_filling.py:71: DeprecationWarning: The 'warn' method is deprecated, use 'warning' instead
  logger.warn("No input files were given: all inputs will be filled with random values!")
[ WARNING ] No input files were given: all inputs will be filled with random values!
[ INFO ] Infer Request 0 filling
[ INFO ] Fill input 'inputs' with random values (image is expected)
[ INFO ] Infer Request 1 filling
[ INFO ] Fill input 'inputs' with random values (image is expected)
[ INFO ] Infer Request 2 filling
[ INFO ] Fill input 'inputs' with random values (image is expected)
[ INFO ] Infer Request 3 filling
[ INFO ] Fill input 'inputs' with random values (image is expected)
[Step 10/11] Measuring performance (Start inference asyncronously, 4 inference requests using 4 streams for CPU, limits: 60000 ms duration)
[Step 11/11] Dumping statistics report
Count:      156 iterations
Duration:   62618.97 ms
Latency:    1587.50 ms
Throughput: 2.49 FPS


python3 /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py -m yolov4-relu/frozen_darknet_yolov4-leaky-416_FP16_model.xml


(tf1.14_venv) dlcv@dlcv-NUC7i7BNH:~/OpenVINO-YOLOV4$ python3 /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py -m frozen_darknet_yolov4-mish-416_FP16_model.xml
[Step 1/11] Parsing and validating input arguments
/opt/intel/openvino_2020.4.287/python/python3.6/openvino/tools/benchmark/main.py:29: DeprecationWarning: The 'warn' method is deprecated, use 'warning' instead
  logger.warn(" -nstreams default value is determined automatically for a device. "
[ WARNING ]  -nstreams default value is determined automatically for a device. Although the automatic selection usually provides a reasonable performance, but it still may be non-optimal for some cases, for more information look at README. 
[Step 2/11] Loading Inference Engine
[ INFO ] InferenceEngine:
         API version............. 2.1.2020.4.0-359-21e092122f4-releases/2020/4
[ INFO ] Device info
         CPU
         MKLDNNPlugin............ version 2.1
         Build................... 2020.4.0-359-21e092122f4-releases/2020/4

[Step 3/11] Setting device configuration
[ WARNING ] -nstreams default value is determined automatically for CPU device. Although the automatic selection usually provides a reasonable performance,but it still may be non-optimal for some cases, for more information look at README.
[Step 4/11] Reading the Intermediate Representation network
[ INFO ] Read network took 114.50 ms
[Step 5/11] Resizing network to match image sizes and given batch
[ INFO ] Network batch size: 1
[Step 6/11] Configuring input of the model
[Step 7/11] Loading the model to the device
[ INFO ] Load network took 2121.87 ms
[Step 8/11] Setting optimal runtime parameters
[Step 9/11] Creating infer requests and filling input blobs with images
[ INFO ] Network input 'inputs' precision U8, dimensions (NCHW): 1 3 416 416
/opt/intel/openvino_2020.4.287/python/python3.6/openvino/tools/benchmark/utils/inputs_filling.py:71: DeprecationWarning: The 'warn' method is deprecated, use 'warning' instead
  logger.warn("No input files were given: all inputs will be filled with random values!")
[ WARNING ] No input files were given: all inputs will be filled with random values!
[ INFO ] Infer Request 0 filling
[ INFO ] Fill input 'inputs' with random values (image is expected)
[ INFO ] Infer Request 1 filling
[ INFO ] Fill input 'inputs' with random values (image is expected)
[ INFO ] Infer Request 2 filling
[ INFO ] Fill input 'inputs' with random values (image is expected)
[ INFO ] Infer Request 3 filling
[ INFO ] Fill input 'inputs' with random values (image is expected)
[Step 10/11] Measuring performance (Start inference asyncronously, 4 inference requests using 4 streams for CPU, limits: 60000 ms duration)
[Step 11/11] Dumping statistics report
Count:      104 iterations
Duration:   62287.86 ms
Latency:    2391.02 ms
Throughput: 1.67 FPS
