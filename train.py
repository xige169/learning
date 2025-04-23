import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
from module import*

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_dataset=torchvision.datasets.CIFAR10('./dataset2', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset=torchvision.datasets.CIFAR10('./dataset2', train=False, transform=torchvision.transforms.ToTensor(), download=True)


train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print(f'训练数据集的长度为: {train_data_size}')
print(f'测试数据集的长度为: {test_data_size}')

train_load=DataLoader(train_dataset,batch_size=64)
test_load=DataLoader(test_dataset,batch_size=64)



# for data in train_load:
#     print(data)

tudui = Tudui()
tudui = tudui.to(device)
#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

#训练次数
train_step=0
#测试次数
test_step=0
#训练轮数
epoch=10
#训练速度
learning_rate=0.01

#优化器
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)
writer = SummaryWriter('log8')

for i in range(epoch):
    print(f'----------第{i+1}轮训练开始----------')
    start_time = time.time()
    tudui.train()
    for data in train_load:
        img,target=data
        img = img.to(device)
        target = target.to(device)

        output=tudui(img)
        loss = loss_fn(output,target)

        optimizer.zero_grad()
        loss.backward()#反向传播
        optimizer.step()

        train_step+=1
        if train_step % 100 ==0:
            print(f'训练次数: {train_step}, loss: {loss}')
            writer.add_scalar('train_loss', loss.item(), train_step)
    end_time = time.time()
    print(f'本轮训练所需时间: {end_time-start_time}')

    tudui.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_load:
            img,target = data
            img = img.to(device)
            target = target.to(device)
            output=tudui(img)
            loss = loss_fn(output,target)
            total_test_loss+=loss.item()
            accuracy=(output.argmax(1) == target).sum()
            total_accuracy+=accuracy
    print(f'整体测试上的loss: {total_test_loss}')
    print(f'模型整体上的正确率: {total_accuracy/test_data_size}')

    writer.add_scalar('test_loss', total_test_loss, test_step)
    writer.add_scalar('accuracy', total_accuracy/test_data_size, test_step)

    test_step+=1

    torch.save(tudui,f'tudui_{i}.pth')
    print('模型已保存')



writer.close()

