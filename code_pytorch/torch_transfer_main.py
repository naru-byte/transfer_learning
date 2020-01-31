import os
import torch
import datasets
from datasets import train_transform, valid_transform, make_loader
import torch_models

base_dir = os.path.expanduser("~/Caltech101")
log_dir = os.path.join(base_dir,"log")
model_name = "Caltech_transfer"

os.makedirs(os.path.join(log_dir,model_name), exist_ok=True)

data_dir = os.path.join(*[base_dir,"data","101_ObjectCategories"])
save_file = "testorch.pth"

x_size = 256
y_size = 256
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

n_classes = 102

datasize = {"train":28, "valid":2}
n_epoch = 50
train_batch, valid_batch = 64 ,32

block = [1,2,3,4,5][1]
fc_shapes = [256]

fine_tuning = [True,False][0]

if __name__ == "__main__":
    #data 
    datas = datasets.data_path(data_dir=data_dir,
                               datasize=datasize)
    train_data, valid_data, test_data = datas()

    train_dataset = datasets.NStrainimageData(image_path_list=train_data[0],
                                              label_list=train_data[1],
                                              transform=train_transform(x_size=x_size, y_size=y_size, mean=mean, std=std))
    
    valid_dataset = datasets.NSvalidationimageData(image_path_list=valid_data[0],
                                                   label_list=valid_data[1],
                                                   transform=valid_transform(x_size=x_size, y_size=y_size, mean=mean, std=std))

    test_dataset = datasets.NSvalidationimageData(image_path_list=test_data[0],
                                                  label_list=test_data[1],
                                                  transform=valid_transform(x_size=x_size, y_size=y_size, mean=mean, std=std))


    train_loader = make_loader(dataset=train_dataset,
                               batch_size=train_batch)

    valid_loader = make_loader(dataset=valid_dataset,
                               batch_size=valid_batch)

    test_loader = make_loader(dataset=test_dataset,
                              batch_size=valid_batch)

    #model
    net = torch_models.Network(block=block,
                               input_shape=(3, y_size, x_size),
                               fc_shapes=fc_shapes,
                               n_classes=n_classes)

    net.train()

    update_params = [] if not fine_tuning else net.parameters()

    if not fine_tuning:
        for name, param in net.named_parameters():
            if "base_vgg" in name:
                param.requires_grad = False
            else:
                update_params.append(param)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=update_params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net.to(device)

    for epoch in range(n_epoch):
        print("-- epoch:{} --".format(epoch))
        net.train()
        train_loss, train_acc = 0.0, 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                _, pred = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                train_acc  += torch.sum(pred == labels.data)

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        print("training_loss : {}".format(train_loss))
        print("training_acc : {}".format(train_acc))

        torch.save(net.state_dict(), os.path.join(*[log_dir,model_name,save_file]))

        net.eval()
        valid_loss, valid_acc = 0.0, 0.0

        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                _, pred = torch.max(outputs, 1)

                valid_loss += loss.item() * inputs.size(0)
                valid_acc  += torch.sum(pred == labels.data)

        valid_loss /= len(valid_loader.dataset)
        valid_acc /= len(valid_loader.dataset)

        print("validation_loss : {}".format(valid_loss))
        print("validation_acc : {}".format(valid_acc))

    test_loss, test_acc = 0.0, 0.0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, pred = torch.max(outputs, 1)

        test_loss += loss.item() * inputs.size(0)
        test_acc  += torch.sum(pred == labels.data)
    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)

    print("test_loss : {}".format(test_loss))
    print("test_acc : {}".format(test_acc))