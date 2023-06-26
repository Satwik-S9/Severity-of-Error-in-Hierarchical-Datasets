#! /bin/bash

echo -e "\nRUNNING THE EXPERIMENTS for NO-CRM"

python /workspace/utsa/MTP-SatwikSrivastava/avg_severity.py --model densenet121 --batch-size 32
python /workspace/utsa/MTP-SatwikSrivastava/avg_severity.py --model resnet18 --batch-size 32 
python /workspace/utsa/MTP-SatwikSrivastava/avg_severity.py --model resnet50 --batch-size 32  
python /workspace/utsa/MTP-SatwikSrivastava/avg_severity.py --model mobilenet --batch-size 32  
python /workspace/utsa/MTP-SatwikSrivastava/avg_severity.py --model wideresnet --batch-size 32  
python /workspace/utsa/MTP-SatwikSrivastava/avg_severity.py --model shufflenet --batch-size 32  
python /workspace/utsa/MTP-SatwikSrivastava/avg_severity.py --model effnetb4 --batch-size 32  