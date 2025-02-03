import os
import tools
import torch
import numpy as np
import warnings
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import shutil
import myUNF
import torch.nn.init as init
from PIL import Image


def postprocess(image):
    print(image.shape)
    # Find contours
    contours_first, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for con in contours_first:
        area = cv2.contourArea(con)
        if area>150:
            contours.append(con)

    output = cv2.drawContours(np.zeros((512,512,3)), contours, -1, (255,255,255), thickness=cv2.FILLED)
    #cv2.imwrite('OUT_{}'.format(id), output)
    # Show result
    #cv2.imshow('Contours', output)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Create a mask from contours

    # Smooth the mask
    blurred_mask = cv2.GaussianBlur(output, (25, 25), 0)

    # Threshold back to binary
    _, smoothed_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

    # Extract smoothed contours
#    contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imwrite('SMOOTH_{}'.format(id), smoothed_mask)
    return smoothed_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('ok')

net = myUNF.UNetFormer(num_classes=2)
#net.decoder.segmentation_head[2] = torch.nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)

#print(net.decoder)
#model_weights = torch.load('./saved_models_0/net_19.pt')
#model_weights = torch.load('./saved_models_1/net_26.pt')
model_weights = torch.load('./clouds_pretrainedonbuilds_saved_models/net_18.pt')

net.load_state_dict(model_weights)

net.to(device)
net.eval()
ids = os.listdir('/home/mariapap/DATASETS/cloud11/TEST/images/')


data_path = '/home/mariapap/DATASETS/cloud11/PATCHES/'

save_folder = 'PREDS' #where to save the models and training progress
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)


def normalize_img(img, mean=[118.974, 128.536, 125.832], std=[78.089, 78.305, 82.140]):  #

    img_array = np.asarray(img, dtype=np.uint8)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img


cm_total = np.zeros((2,2))
for id in ids:
    img = Image.open('/home/mariapap/DATASETS/cloud11/TEST/images/{}'.format(id))

    label = Image.open('/home/mariapap/DATASETS/cloud11/TEST/labels/{}'.format(id))
    label = np.array(label)/255.

    img = np.array(img)
    img = normalize_img(img)
    img = torch.from_numpy(img).to(device)
    img = img.permute(2,0,1).unsqueeze(0)
    pred = net(img)

    label_conf, pred_conf = label.flatten(), torch.argmax(torch.softmax(pred, 1),1).data.cpu().numpy()


    #cm = confusion_matrix(label_conf, pred_conf.flatten(), labels=[0, 1])
    #cm_total += cm
    s_pred = np.array(pred_conf[0]*255, dtype=np.uint8)

    s_pred = postprocess(s_pred)
    s_pred = np.array(s_pred[:,:], dtype=np.uint8)
    s_pred = Image.fromarray(s_pred)
    s_pred.save('./{}/{}'.format(save_folder, id))

print('Conf Matrix')
print(cm_total)
