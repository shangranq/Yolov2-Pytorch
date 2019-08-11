import torch
import torch.nn as nn

class Yolo_v2(nn.Module):
    def __init__(self, nb_box, nb_class, feature_extractor):
        super(Yolo_v2, self).__init__()
        self.nb_box = nb_box
        self.nb_class = nb_class
        if feature_extractor == 'MobileNet':
            self.feature_extractor = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True).features
        else:
            raise Exception('Architecture not supported! Only support MobileNet at the moment!')
        self.detect_layer = nn.Conv2d(1280, self.nb_box * (4 + 1 + self.nb_class),
                                      kernel_size=(1, 1), stride=(1, 1), padding=0)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.detect_layer(x)
        x = x.view(-1, self.nb_box, (4 + 1 + self.nb_class), 13, 13)
        return x

if __name__ == "__main__":
    Yolo = Yolo_v2(5, 80, "MobileNet")            
    test_input = torch.Tensor(1, 3, 416, 416)
    output = Yolo(test_input)
    print(output.shape)




