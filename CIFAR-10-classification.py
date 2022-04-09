
from keras.datasets import cifar10
from alexnet_pytorch import AlexNet
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import cm
import numpy as np
import pickle
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
labels = {
    0:"airplane",
    1:"automobile",
    2:"bird",
    3:"cat",
    4:"deer",
    5:"dog",
    6:"frog",
    7:"horse",
    8:"ship",
    9:"truck",
}
alexNet_labels = {

}
x_valid = x_train[40000:50000,:]
y_valid = y_train[40000:50000,:]
x_train = x_train[0:40000,:]
y_train = y_train[0:40000,:]
# reshaping data to 0-1 range
# x_train = x_train / 255.0
# x_test = x_test / 255.0

#preprocess image function
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#use alexnet
model = AlexNet.from_pretrained('alexnet')
model.eval()


#read the alexnet for all the x_train saved into pkl
# for  index, img in enumerate(x_train):
#   print(index)
#   img0 = Image.fromarray(img)
#   img_tensor = preprocess(img0)
#   batch = img_tensor.unsqueeze(0)

#   # use gpu if available
#   if torch.cuda.is_available():
#       batch = batch.to("cuda")
#       model.to("cuda")

#   with torch.no_grad():
#     outputs = model(batch)

#   probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
#   with open ("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
#   all_probs, respective_cats = torch.topk(probabilities, 1000)
#   selected_cat = categories[respective_cats[np.argmax(all_probs)]]
  
#   if(selected_cat in alexNet_labels):
#     alexNet_labels[selected_cat][y_train[index][0]] += 1
#   else:
#     alexNet_labels[selected_cat] = np.zeros(10)
#     alexNet_labels[selected_cat][y_train[index][0]] += 1
# print(alexNet_labels)
# with open("alexNet_labels_dict_on_trainXx.pkl", 'wb') as outp:  # Overwrites any existing file.
#   pickle.dump(alexNet_labels, outp, pickle.HIGHEST_PROTOCOL)

# once data is trained with the code above, filter top 10, and reorganize into table
# with open('alexNet_labels_dict_on_trainX.pkl', 'rb') as inp:
#   alexNet_labels_pickle = pickle.load(inp)

# alexNet_labels_total = {}
# #picking the 10 most frequent categories
# for cat in alexNet_labels_pickle:
#   alexNet_labels_total[cat] = sum(alexNet_labels_pickle[cat])

# top_10 = sorted(alexNet_labels_total, key=alexNet_labels_total.get, reverse=True)[:10]

# confusion_matrix = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
# x_labels = np.array([""])

# #making matrix
# np.set_printoptions(suppress=True)
# for cat in top_10:
#   x_labels = np.vstack((x_labels, cat))
#   confusion_matrix = np.vstack((confusion_matrix, alexNet_labels_pickle[cat]))
# confusion_matrix = np.hstack((x_labels, confusion_matrix))

# #print pretty like
# s = [[str(e) for e in row] for row in confusion_matrix]
# lens = [max(map(len, col)) for col in zip(*s)]
# fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
# table = [fmt.format(*row) for row in s]
# print ('\n'.join(table))

# d)

#following block of comment extracts the feature
# x_test_features= torch.tensor([])
# def cross_entropy_loss(y_one_hot, activations):
#     return -torch.mean(  # 3
#         torch.sum(  # 2
#             y_one_hot * torch.log(activations), axis=1  # 1
#         )
#     )
# fc7 = nn.Sequential(*list(model.classifier.children())[:-2])
# fc7.eval()
# print(nn.Sequential(*list(model.classifier.children())))
# print((model))
# for  index, img in enumerate(x_test): #change to x_train or x_text depending on what features you are extracting
#   print(index)
#   img0 = Image.fromarray(img)
#   img_tensor = preprocess(img0)
#   batch = img_tensor.unsqueeze(0)

#   # use gpu if available
#   if torch.cuda.is_available():
#       batch = batch.to("cuda")
#       model.to("cuda")

#   with torch.no_grad():
#     outputs = model.extract_features(batch)
#     outputs = torch.reshape(outputs, (1, 9216))
#     fc7_outputs = fc7.forward(outputs)
#   x_test_features = torch.cat((x_test_features,fc7_outputs), 0)
  
# print(x_test_features)
# with open("x_test_features.pkl", 'wb') as outp:  # Overwrites any existing file.
#   pickle.dump(x_test_features, outp, pickle.HIGHEST_PROTOCOL)

#opening the pickled files
# with open('x_train_features_fc7.pkl', 'rb') as inp:
#   x_train_features = pickle.load(inp)
# with open('x_test_features_fc7.pkl', 'rb') as inp:
#   x_test_features = pickle.load(inp)
# y_train_tensor = torch.flatten(torch.tensor(y_train))
# y_test_tensor = torch.flatten(torch.tensor(y_test))

# x_train_features = torch.relu(x_train_features)
# x_test_features = torch.relu(x_test_features)

# print(x_train_features)



# # # logistic regression

# def one_hot_encode(vector):
#     n_classes = len(vector.unique())  
#     one_hot = torch.zeros((vector.shape[0], n_classes))\
#         .type(torch.LongTensor)  
#     return one_hot\
#         .scatter(1, vector.type(torch.LongTensor).unsqueeze(1), 1)  
# y_train_onehot = one_hot_encode(y_train_tensor)
# y_test_onehot = one_hot_encode(y_test_tensor)
# w = torch.rand((4096, 10))
# b = torch.rand(10)

# loss_func = torch.nn.CrossEntropyLoss()

# lambda_param = 0.01
# learning_rate = 0.1

# iterations = 200
# for i in range(0, iterations):
#     Z = torch.mm(x_train_features, w) + b
#     A = torch.softmax(Z, dim=1)
#     l2_regularization = torch.sum(w ** 2)
#     loss = loss_func(A, y_train_tensor) + lambda_param * l2_regularization
#     w_gradients = -torch\
#         .mm(x_train_features.transpose(0, 1), y_train_onehot - A) / x_train_features.shape[0] \
#                   + (2 * lambda_param * w)
#     b_gradients = -torch.mean(y_train_onehot - A, axis=0)

#     w -= learning_rate * w_gradients
#     b -= learning_rate * b_gradients

#     if i % 2 == 0:
#         print("Loss at iteration {}: {}".format(i, loss))

# test_predictions = torch.argmax(
#     torch.softmax(torch.mm(x_test_features, w) + b, dim=1), axis=1
# )
# test_accuracy = float(sum(test_predictions == y_test_tensor)) / y_test_tensor.shape[0]
# print("\nFinal Test Accuracy: {}".format(test_accuracy))

# # logistic regression with pytorch nn for fun
# logistic_model = torch.nn.Sequential(
#   torch.nn.Linear(4096, 1024, bias=True),
#   torch.nn.ReLU(inplace=True),
#   torch.nn.Linear(1024, 10, bias=True),
#   )
# learning_rate = 0.1
# lambda_param = 0.01
# optimizer = torch.optim.SGD(
#   logistic_model.parameters(),
#   lr=0.1,
#   weight_decay=lambda_param
# )
# loss_function = torch.nn.CrossEntropyLoss()

# iterations = 80
# for i in range(0, iterations):
#   Z = logistic_model(x_train_features)
#   loss = loss_function(Z, torch.flatten(y_train_tensor))
#   optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()
#   if(i % 2 == 0):
#     print("Loss at iteration {}: {}".format(i, loss))
#   test_predictions = torch.argmax(
#     torch.softmax(logistic_model(x_test_features), 1), axis=1
#   )
#   test_accuracy = float(sum(test_predictions == y_test_tensor)) / y_test_tensor.shape[0]
#   print("\nFinal Test Accuracy: {}".format(test_accuracy))


# c)

#following block of comment extracts the feature for fc6
x_train_features= torch.tensor([])
x_test_features = torch.tensor([])
def cross_entropy_loss(y_one_hot, activations):
    return -torch.mean(  # 3
        torch.sum(  # 2
            y_one_hot * torch.log(activations), axis=1  # 1
        )
    )
fc6 = nn.Sequential(*list(model.classifier.children())[:-5])
fc6.eval()
print(*list(model.classifier.children())[:-5])
print((model))
print(x_train.shape)
for  index, img in enumerate(x_train):
  print(index)
  img0 = Image.fromarray(img)
  img_tensor = preprocess(img0)
  batch = img_tensor.unsqueeze(0)

  # use gpu if available
  if torch.cuda.is_available():
      batch = batch.to("cuda")
      model.to("cuda")

  with torch.no_grad():
    outputs = model.extract_features(batch)
    outputs = torch.reshape(outputs, (1, 9216))
    fc6_outputs = fc6.forward(outputs)
  x_train_features = torch.cat((x_train_features,fc6_outputs), 0)
  
print(x_train_features.shape)
with open("x_train_features_fc6.pkl", 'wb') as outp:  # Overwrites any existing file.
  pickle.dump(x_train_features, outp, pickle.HIGHEST_PROTOCOL)

print(x_test.shape)
for  index, img in enumerate(x_test):
  print(index)
  img0 = Image.fromarray(img)
  img_tensor = preprocess(img0)
  batch = img_tensor.unsqueeze(0)

  # use gpu if available
  if torch.cuda.is_available():
      batch = batch.to("cuda")
      model.to("cuda")

  with torch.no_grad():
    outputs = model.extract_features(batch)
    outputs = torch.reshape(outputs, (1, 9216))
    fc6_outputs = fc6.forward(outputs)
  x_test_features = torch.cat((x_test_features,fc6_outputs), 0)
  
print(x_test_features)
with open("x_test_features_fc6.pkl", 'wb') as outp:  # Overwrites any existing file.
  pickle.dump(x_test_features, outp, pickle.HIGHEST_PROTOCOL)

# opening the pickled files
with open('x_train_features_fc6.pkl', 'rb') as inp:
  x_train_features = pickle.load(inp)
with open('x_test_features_fc6.pkl', 'rb') as inp:
  x_test_features = pickle.load(inp)
y_train_tensor = torch.flatten(torch.tensor(y_train))
y_test_tensor = torch.flatten(torch.tensor(y_test))
print(x_train_features.shape)
print(x_test_features.shape)