import torch
# 评估测试集准确度代码
def accuracy(data_iter, net):
    acc_num = 0
    total_num = 0
    for X, y in data_iter:
        X = X.cuda()
        y = y.cuda()
        y_pred = net(X)
        acc_num += (y_pred.argmax(dim=1) == y).float().sum().item()
        total_num += y.shape[0]
    return acc_num / total_num

# 训练模型代码
def train(net, train_iter, num_epoch, loss_fn, optimizer, test_iter=None):
    for epoch in range(num_epoch):
        train_loss, train_acc, total_num = 0, 0, 0
        for X, y in train_iter:
            X = X.cuda()
            y = y.cuda()
            y_pred = net(X)
            loss = loss_fn(y_pred, y).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (y_pred.argmax(dim=1) == y).float().sum().item()
            total_num += y.shape[0]
        if test_iter is not None:
            test_acc = accuracy(test_iter, net)
            print('当前第{}轮 训练集：loss = {:.4f}, acc = {:.4f}  测试集：acc = {:.4f}'.format(epoch+1, train_loss/total_num, train_acc/total_num, test_acc))
        else:
            print('当前第{}轮 训练集：loss = {:.4f}, acc = {:.4f}'.format(epoch+1, train_loss/total_num, train_acc/total_num))
