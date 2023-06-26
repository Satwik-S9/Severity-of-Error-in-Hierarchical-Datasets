#! /bin/bash

echo -e "\n##### THIS SCRIPT CALCULATES THE MULTILABEL-SEVERITIES FOR THE MODELS #####\n"

echo -e "\nCalculating severity for DenseNet121 ...\n"
python /workspace/utsa/MTP-SatwikSrivastava/tests/severity.py --model densenet121 --lb

echo -e "\nCalculating severity for ResNet18 ...\n"
python /workspace/utsa/MTP-SatwikSrivastava/tests/severity.py --model resnet18 --lb

echo -e "\nCalculating severity for ResNet50 ...\n"
python /workspace/utsa/MTP-SatwikSrivastava/tests/severity.py --model resnet50 --lb

echo -e "\nCalculating severity for MobileNet ...\n"
python /workspace/utsa/MTP-SatwikSrivastava/tests/severity.py --model mobilenet --lb

echo -e "\nCalculating severity for ShuffleNet ...\n"
python /workspace/utsa/MTP-SatwikSrivastava/tests/severity.py --model shufflenet --lb

echo -e "\nCalculating severity for WideResNet50 ...\n"
python /workspace/utsa/MTP-SatwikSrivastava/tests/severity.py --model wideresnet --lb

echo -e "\nCalculating severity for EfficientNetB4 ...\n"
python /workspace/utsa/MTP-SatwikSrivastava/tests/severity.py --model effnetb4 --lb