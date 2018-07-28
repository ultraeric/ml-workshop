import torchvision.models as models
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import transforms
import cv2

batch_size = 16
epochs = 5
trans = transforms.Compose([transforms.RandomRotation(30),
                            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                            transforms.ToTensor()])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('./data', trans),
    batch_size=batch_size,
    shuffle=True
)

model = models.squeezenet1_1(pretrained=True)
# model = torch.load('gesture_classifier.model')
params = model.parameters()
optimizer = optim.SGD(params, lr=1e-3, momentum=0.9)
loss_func = torch.nn.BCEWithLogitsLoss()

for _ in range(epochs):
    for i, (x, y) in enumerate(train_loader):
        pred = (model(x)[:,0] - 0.5)
        loss = loss_func(pred, y.float())
        loss.backward()
        optimizer.step()
        model.zero_grad()
        print(loss.item())
torch.save(model, 'gesture_classifier.model')
model = torch.load('gesture_classifier.model')

window_name = 'Display'

cv2.namedWindow(window_name)
vc = cv2.VideoCapture(0)
if vc.isOpened(): # try to get the first frame
    exists, frame = vc.read()
else:
    exists = False

i = 0
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 1
fontColor = (255,255,255)
lineType = 2
model.eval()
while exists:
    start_x = (frame.shape[1] - frame.shape[0]) // 2
    end_x = frame.shape[1] - start_x
    frame = frame[:,start_x:end_x,:]
    small_frame = torch.Tensor(cv2.resize(frame, (224, 224))).permute(2, 0, 1).unsqueeze(0)
    pred = (model(small_frame)[0][0] - 0.5).sigmoid()
    print(pred.detach().numpy())
    pred = 1 if pred.detach().numpy() > 0.5 else 0
    cv2.putText(frame,
                'Predicted: {}'.format(pred),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(20)
    if key == 27:
        break
    exists, frame = vc.read()
    i += 1

cv2.destroyWindow(window_name)