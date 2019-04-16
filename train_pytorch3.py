from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os

#import skimage.io
import PIL.Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from vocparseclslabels import PascalVOC

import sklearn.metrics

all_data = {}
all_labels = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']


# freeze feature to freeze starting layers from being re-trained
def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


class dataset_voc(Dataset):
  def __init__(self, root_dir, trvaltest, transform=None):

    """
    Args:

        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

    self.root_dir = root_dir

    self.transform = transform
    self.imgfilenames=[]
    self.labels=[]

    pv=PascalVOC(root_dir) # './data/VOCdevkit/VOC2012/'
    cls=pv.list_image_sets()
    #self.classlist=cls

    if trvaltest==0:
      dataset='train'
    elif trvaltest==1:
      dataset='val'
    else:
      print('bla!!unsupported!')
      exit()

    
    filenamedict={}
    for c,cat_name in enumerate(cls):
      imgstubs=pv.imgs_from_category_as_list(cat_name, dataset)
      for st in imgstubs:
        #fn=os.path.join(self.root_dir,'JPEGImages',st+'.jpg')
        if st in filenamedict:
          filenamedict[st][c]=1
        else:
          vals=np.zeros(20,dtype=np.int32)
          vals[c]=1
          filenamedict[st]=vals


    self.labels=np.zeros((len(filenamedict),20))
    tmpct=-1
    for key,value in filenamedict.items():
      tmpct+=1
      self.labels[tmpct,:]=value

      fn=os.path.join(self.root_dir,'JPEGImages',key+'.jpg')
      self.imgfilenames.append(fn)
        #self.imgfilenames.append(fn)
        #self.labels.append(int(c))

    # for i,fn in enumerate(self.imgfilenames):
      # print(fn)
      # print(self.labels[i,:].sum())

    print('dataset init done')

  def __len__(self):
      return len(self.imgfilenames)

  def __getitem__(self, idx):
    image = PIL.Image.open(self.imgfilenames[idx])
    #print('img native', image.size)

    label=self.labels[idx,:].astype(np.float32)

    #print(self.imgfilenames[idx])
    if self.transform:
      image = self.transform(image)

    # if you do five crop, thne you MUST change this part, as outputs are now 4 d tensors!!!
    if image.size()[0]==1:
      image=image.repeat([3,1,1])
    #print(image.size())

    sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}

    return sample


def train_model3(dataloaders, dataset_sizes, model,  optimizer, scheduler, num_epochs=25, use_gpu=False, classweights = None, posweights= None):
    since = time.time()

    best_model_wts = model.state_dict()
    best_prec = 0.0
    best_acc = 0.0

    # freeze model layers
    freeze_layer(model.conv1)
    freeze_layer(model.bn1)
    freeze_layer(model.layer1)
    freeze_layer(model.layer2)

    numcl=20
    criterion=torch.nn.BCEWithLogitsLoss(weight=classweights, pos_weight=posweights) #, size_average=True, reduce=True)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = np.zeros(numcl)
            clscounts=np.zeros(numcl)

            allpreds=[[] for _ in range(numcl)]
            alllb=[[] for _ in range(numcl)]
            avgprecs=np.zeros(numcl)

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                #inputs, labels, filenames = data
                inputs=data['image']
                clabels=data['label']    

                #print(type(clabels))
                #print('type(inputs)',type(inputs))  
                #print(inputs.shape)      
                
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(clabels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(clabels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                preds = outputs.data>=0

                loss = criterion(outputs, labels)
                print(loss)
                # loss=0
                # criterion = torch.nn.BCEWithLogitsLoss(weight=None, pos_weight=posweights)
                # for c in range(numcl):
                #   loss+=classweights[:,c]*criterion(outputs[:,c],labels[:,c])


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.cpu().item() #.data[0]
                for c in range(numcl):
                  running_corrects[c]+=torch.sum(preds.cpu()[:,c] == clabels.type(torch.ByteTensor)[:,c])
                  clscounts[c]+=torch.sum(clabels[:,c])

                  allpreds[c].extend(preds.cpu()[:,c].numpy().tolist())
                  alllb[c].extend(clabels.type(torch.ByteTensor)[:,c].numpy().tolist())


                # todo: record into all_data while in validation
                if phase == 'val':
                    for i in range(len(inputs)):
                        for j in range(20):
                            if clabels[i][j] == 1.:
                                if all_labels[j] not in all_data:
                                    all_data[all_labels[j]] = []
                                all_data[all_labels[j]].append({
                                    'image': data['filename'][i],
                                    'score': 0,
                                })

                #running_corrects += torch.sum(preds == labels.data).cpu() #.numpy()

            epoch_loss = running_loss / float(dataset_sizes[phase])
            epoch_acc=0.
            avgprec=0.
            for c in range(numcl):
              epoch_acc+=running_corrects[c]/float(dataset_sizes[phase])/20.0

              avgprecs[c]=sklearn.metrics.average_precision_score(alllb[c], allpreds[c], average='macro', sample_weight=None)
              print(c,running_corrects[c]/float(dataset_sizes[phase]),avgprecs[c])


            #epoch_acc = running_corrects / float(dataset_sizes[phase])
            #print(type(epoch_acc),type(np.mean(avgprecs)),np.mean(avgprecs))
            
            #print('{} Loss: {:.4f} Acc: {:.4f}, avgprec {:.4f} '.format(
            #    phase, epoch_loss, epoch_acc, str(np.mean(avgprecs))))

            print(phase, 'loss', epoch_loss ,'acc' ,epoch_acc, 'avgprec', str(np.mean(avgprecs)) )

            # deep copy the model
            if phase == 'val' and np.mean(avgprecs) > best_prec:
                print('better model')
                best_prec = np.mean(avgprecs)
                best_model_wts = model.state_dict()



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

###############################

def runstuff():

  numcl=20
  #transforms
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

  image_datasets={}
  image_datasets['train']=dataset_voc(root_dir='./static/VOCdevkit/VOC2012/',trvaltest=0, transform=data_transforms['train'])
  image_datasets['val']=dataset_voc(root_dir='./static/VOCdevkit/VOC2012/',trvaltest=1, transform=data_transforms['val'])

  dataloaders = {di: torch.utils.data.DataLoader(image_datasets[di], batch_size=32, shuffle=True, num_workers=4)  
                    for di in ['train', 'val'] }
  dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
  #class_names = image_datasets['train'].classes

  use_gpu = torch.cuda.is_available()



  #model
  model_ft = models.resnet18(pretrained=True)
  num_ftrs = model_ft.fc.in_features
  model_ft.fc = nn.Linear(num_ftrs, numcl)



  if use_gpu:
      model_ft = model_ft.cuda(0)


  # Observe that all parameters are being optimized
  optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

  # Decay LR by a factor of 0.1 every ? epochs
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

  clweights = torch.unsqueeze(torch.from_numpy(len(image_datasets['train'])/image_datasets['train'].labels.sum(axis=0)),0).type(torch.FloatTensor)
  if use_gpu:
      clweights = clweights.to('cuda')
  model_ft = train_model3(dataloaders,dataset_sizes, model_ft, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25, use_gpu=use_gpu, classweights=clweights)

  torch.save(model_ft.state_dict(), "./raw_model.pt")


if __name__=='__main__':
  #tester2()
  #tester3()
  runstuff()


