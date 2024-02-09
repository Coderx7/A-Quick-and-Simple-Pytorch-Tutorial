#%%
# simplenet_cifar_5m_extra_pool
import sys
print(sys.version)
import torch
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from simplenet import SimpleNet, simplenet_cifar_5m_extra_pool, simplenet_cifar_5m
from simplenet2 import simplenet
print(f'{torch.__version__}')


model0 = simplenet()
model = simplenet_cifar_5m_extra_pool(num_classes=10)

print(f'{model}')
print(f'{model0}')
state_dict = torch.load("/media/hossein/SSD1/code_dl/best_chkpt_simplenet_cifar10_2018-12-25_00-34-57.pth.tar", map_location=lambda storage, loc: storage)['state_dict']
# Create a new state dictionary without the "module." prefix
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
print('after purge:')
# remove the dorpout layers otherthan the ones after maxpools 
def remove_dropout_layers(model):
    features = torch.nn.Sequential()
    prev_layer = None
    i=0
    for layer in model.features.children():
        if isinstance(layer, torch.nn.Dropout2d):
            if not isinstance(prev_layer, torch.nn.MaxPool2d):
                continue
            layer = torch.nn.Dropout2d(0.1)
        prev_layer = layer
        features.add_module(str(i), layer);i+=1
    # assign the updated features
    model.features = features
    return model

model = remove_dropout_layers(model)
# load the new state_dict into the model
model.load_state_dict(state_dict)

