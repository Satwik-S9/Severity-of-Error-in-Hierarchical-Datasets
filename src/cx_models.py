import torch
import torch.nn as nn
import torchvision.models as tvm
from types import SimpleNamespace
from .efficientnet import construct_model

class DenseNet121(nn.Module):
    def __init__(self, num_classes: int, pre_trained: bool = True, mtype: str = 'local'):
        """ DenseNet121 Model for CheXPert dataset

        Args:
            num_classes (int): Number of classes that it needs to predict.
            pre_trained (bool, optional): use a pre-trained model. Defaults to True.
        """
        super().__init__()
        
        self.mtype = mtype
        # Initialize the network
        if mtype == 'local':
            self.densenet121 = tvm.densenet121(pretrained=pre_trained)
            # change the classifier output
            clf_in = self.densenet121.classifier.in_features
            self.densenet121.classifier = nn.Sequential(nn.Linear(clf_in, num_classes), nn.Sigmoid())

        elif mtype == 'kaggle':
            self.net = tvm.densenet121(pretrained=pre_trained)
            # change the classifier output
            clf_in = self.net.classifier.in_features
            self.net.classifier = nn.Sequential(nn.Linear(clf_in, num_classes), nn.Sigmoid())

        else:
            raise ValueError("Invalid Model Type !!!")
        
        
    def forward(self, input):
        if self.mtype == 'local':
            return self.densenet121(input)
        else:
            return self.net(input)

class Resnet18(nn.Module):
    def __init__(self, num_classes: int, pre_trained: bool = True) -> None:
        """Resnet18 Model for CheXpert data

        Args:
            num_classes (int): _description_
            pre_trained (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        
        self.resnet18 = tvm.resnet18(pretrained=pre_trained)
        in_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(nn.Linear(in_ftrs, num_classes)) # nn.Sigmoid())
        
    def forward(self, x):
        return self.resnet18(x)
    
class RenNet152(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        
        self.resnet152 = tvm.resnet152(pretrained=True)
        in_ftrs = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Linear(in_ftrs, num_classes)
 
class MobileNet(nn.Module):
    def __init__(self, num_classes: int, pretrained=True) -> None:
        super().__init__()
        self.network = tvm.mobilenet_v3_small(pretrained)
        in_ftr = self.network.classifier[3].in_features
        self.network.classifier[3] = nn.Sequential(nn.Linear(in_ftr, num_classes),
                                                   nn.Sigmoid())
        
    def forward(self, x):
        return self.network(x)

        
class Resnet50(nn.Module):
    def __init__(self, num_classes: int, pretrained=True) -> None:
        super().__init__()
        self.network = tvm.resnet50(pretrained)
        in_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(nn.Linear(in_ftrs, num_classes), 
                                        nn.Sigmoid())
        
    def forward(self, x):
        return self.network(x)

        
class GoogleNet(nn.Module):
    def __init__(self, num_classes: int, pretrained=True) -> None:
        super().__init__()
        self.network = tvm.googlenet(pretrained)
        in_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(nn.Linear(in_ftrs, num_classes),
                                        nn.Sigmoid())
        
    def forward(self, x):
        return self.network(x) 
      
class EffnetB4(nn.Module):
    def __init__(self, num_classes: int, pretrained=True) -> None:
        super().__init__()
        self.network = tvm.efficientnet_b4(pretrained)
        in_ftrs = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True),
                                                nn.Linear(in_ftrs, num_classes),
                                                nn.Sigmoid())
        
    def forward(self, x):
        return self.network(x) 

def load_base_classifier(model_name: str, cfg: SimpleNamespace):
    MODEL_NAMES = ['resnet152', 'baseline', 'densenet', 'resnet18', 'resnet', 'densenet121', 'effnetb4']
    if model_name not in MODEL_NAMES:
        raise ValueError('model_name should be in {}'.format(MODEL_NAMES))

    model = None
    if model_name == 'resnet152':
        model = tvm.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 5)
    elif model_name == 'effnetb4':
        model = construct_model('efficientnet-b4', 5)
    else:
        raise NotImplementedError(f"Method Not Implemented for {model_name}")
    
    return model

class Resnet18HXE(nn.Module):
    def __init__(self, num_classes: int, pretrained=True) -> None:
        super().__init__()
        self.network = tvm.resnet18(pretrained)
        in_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(nn.Linear(in_ftrs, num_classes), 
                                        nn.Sigmoid())
    
    def forward(self, x):
        return self.network(x)
    
        
class Densenet121HXE(nn.Module):
    def __init__(self, num_classes: int, pretrained=True) -> None:
        super().__init__()
        self.network = tvm.densenet121(pretrained)
        in_ftrs = self.network.classifier.in_features
        self.network.classifier = nn.Sequential(nn.Linear(in_ftrs, num_classes), 
                                                nn.Sigmoid())
        
    def forward(self, x):
        return self.network(x)

class WideResnet50(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool=True) -> None:
        super().__init__()
        self.network = tvm.wide_resnet50_2(pretrained)
        in_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(nn.Linear(in_ftrs, num_classes), 
                                                nn.Sigmoid())
        
    def forward(self, x):
        return self.network(x)
        
        
class ShuffleNet1_0(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool=True) -> None:
        super().__init__()
        self.network = tvm.shufflenet_v2_x1_0(pretrained)
        in_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(nn.Linear(in_ftrs, num_classes), 
                                                nn.Sigmoid())
        
    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    x = torch.randn((32, 3, 224, 224))
    model = DenseNet121(14)
    out = model(x)
    print(out.size())
    