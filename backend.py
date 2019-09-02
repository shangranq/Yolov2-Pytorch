import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

class Yolo_v2(nn.Module):

    def __init__(self, nb_box, nb_class, feature_extractor):
        entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
        print(entrypoints)

        super(Yolo_v2, self).__init__()
        self.nb_box = nb_box
        self.nb_class = nb_class
        self.feature_extractor = feature_extractor

        if feature_extractor == 'MobileNet':
            self.feature_extractor = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True).features
            self.detect_layer = nn.Conv2d(1280, self.nb_box * (4 + 1 + self.nb_class),
                                          kernel_size=(1, 1), stride=(1, 1), padding=0)

        elif feature_extractor == 'ResNet':
            resnet = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
            self.detect_layer = nn.Conv2d(2048, self.nb_box * (4 + 1 + self.nb_class),
                                          kernel_size=(1, 1), stride=(1, 1), padding=0)
        else:
            raise Exception('Architecture not supported! Only support MobileNet at the moment!')

        new_kernel = np.random.normal(size=self.detect_layer.weight.data.shape) / (13 * 13)
        new_bias = np.random.normal(size=self.detect_layer.bias.data.shape) / (13 * 13)
        self.detect_layer.weight.data = torch.Tensor(new_kernel)
        self.detect_layer.bias.data = torch.Tensor(new_bias)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.detect_layer(x)
        x = x.view(-1, self.nb_box, (4 + 1 + self.nb_class), 13, 13)
        return x

    def normalize(self, image):
        image = image / 255.0
        image = (image - image.mean()) / image.std()
        image[:, :, 0] = image[:, :, 0] * 0.229 + 0.485
        image[:, :, 1] = image[:, :, 1] * 0.224 + 0.456
        image[:, :, 2] = image[:, :, 2] * 0.225 + 0.406
        return image


if __name__ == "__main__":
    Yolo = Yolo_v2(5, 80, "ResNet")
    test_input = torch.Tensor(1, 3, 416, 416)
    output = Yolo(test_input)
    # print(output.shape)
    # for p in Yolo.parameters():
    #     print(p.requires_grad, p.data.shape)




