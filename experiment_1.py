import torch
from torch import optim, nn
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import visdom
import pandas as pd

class Read_data(Dataset):

    def __init__(self, root,  mode):
        """
        :param root: the path of the dataset
        :param resize: the shape of the signal
        :param mode: the use of the dataset (train / validatation / test)
        """
        super(Read_data, self).__init__()

        self.root = root

        # to initialize the label to the signal
        self.name2label = {}
        for name in sorted(os.listdir(root)):
        # os.listdir(): to get the name of the file of the path given
            if not os.path.isdir(os.path.join(root, name)):
            # os.path.isdir(): to decide whether the certain root is a file folder
            # os.path.join( , ): to concatenate the path
                continue

            self.name2label[name] = len(self.name2label.keys())
            # set the label to the data

        # to get the signals and the labels
        self.signals, self.labels = self.load_csv('signals.csv')

        # to split the dataset
        if mode == 'train': # 60%
            self.signals = self.signals[:int(0.6 * len(self.signals))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val': # 20% = 60% -> 80%
            self.signals = self.signals[int(0.6 * len(self.signals)):int(0.8 * len(self.signals))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else: # 20% = 80% -> end
            self.signals = self.signals[int(0.8 * len(self.signals)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
        # os.path.exists(): to check if there is the  certain file
            # to create the csv file
            signals = []
            for name in self.name2label.keys():
                for i in range(51):
                    signals += glob.glob(os.path.join(self.root, name, str(i - 20) + 'dB', '*.csv'))

            random.shuffle(signals)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for sig in signals:
                    name = sig.split(os.sep)[-3]
                    # os.sep: the break like '/' in MAC OS operation system
                    label = self.name2label[name]
                    writer.writerow([sig, label])
                print("write into csv file:", filename)

        # read the csv file
        signals, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                sig, label = row
                label = int(label)
                signals.append(sig)
                labels.append(label)

        assert len(signals) == len(labels)

        return signals, labels

    def __len__(self):
        # this function enable we use len() to get the length of the dataset
        return len(self.signals)

    def __getitem__(self, idx):
        # this function enable we use p[key] to get the value
        if idx < 0 or idx > len(self.signals) - 1:
            print("the idx is wrong!")
            os._exit(1)
        sig, label = self.signals[idx], self.labels[idx]

        data = torch.from_numpy(pd.read_csv(sig).values).float()
        label = torch.tensor(label)

        return data, label

class Net(nn.Module):

    def __init__(self, num_class):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(16)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(32)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(64)
        )
        self.line1 = nn.Linear(64*7, 128)
        self.dp1 = nn.Dropout(0.5)
        self.line2 = nn.Linear(128, 64)
        self.dp2 = nn.Dropout(0.5)
        self.out = nn.Linear(64, num_class)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = x.squeeze()
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # x = F.adaptive_max_pool2d(x, [1, 1])

        # flaten operation
        x = x.view(x.size(0), -1)
        # [b, 32*3*3] => [b, 10]
        x = F.relu(self.dp1(self.line1(x)))
        x = F.relu(self.dp2(self.line2(x)))
        x = self.out(x)

        return x


# build the model
batchsz = 16192
lr = 7.5e-4
epochs = 1000
device = torch.device('cpu')
torch.manual_seed(1234)
# set up dataset object
train_db = Read_data('data', mode='train')
val_db = Read_data('data', mode='val')

# set up a DataLoader object
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, shuffle=False, num_workers=2)

viz = visdom.Visdom()

def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    # print(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.unsqueeze(1)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total


def main():

    model = Net(4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    viz.line([0], [-1], win='loss_2', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc_2', opts=dict(title='val_acc'))
    bes_acc, best_epoch = 0, 0
    global_step_1 = 0
    global_step_2 = 0
    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            model.train()
            x = x.unsqueeze(1)
            # x: [b, 3, 224, 224] y: [b]
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step_1], win='loss_2', update='append')
            global_step_1 += 1

        if epoch % 2 == 0:
            with torch.no_grad():
                model.eval()
                val_acc = evaluate(model, val_loader)
                print('epoch:', epoch, 'val_acc:', val_acc)
                if val_acc > bes_acc:
                    best_epoch = epoch
                    bes_acc = val_acc

                viz.line([val_acc], [global_step_2], win='val_acc_2', update='append')
                global_step_2 += 1

    print('best acc:', bes_acc, 'best epoch:', best_epoch)

if __name__ == '__main__':
    main()