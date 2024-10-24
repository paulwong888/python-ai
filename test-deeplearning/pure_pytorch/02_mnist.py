import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


DOWNLOAD_MNIST = False

class Netork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)

def get_data_set(train : bool):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    return datasets.MNIST(
        root = './mnist',
        train = train,
        transform = transform,
        download = DOWNLOAD_MNIST
    )

def train_model():
    # train_dataset = datasets.MNIST.
    train_dataset = get_data_set(True)

    print("tran_data_set length: " + str(len(train_dataset)))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )

    print("train loader length: " + str(len(train_loader)))

    for batch_idx, (data, label) in enumerate(train_loader):
        if batch_idx == 3:
            break

        print(f"batch_idx: {batch_idx}, data: {data.shape}, label: {label.shape}")

    model = Netork()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        for batch_idx, (data, label) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/10 "
                    f"| Batch {batch_idx}/{len(train_loader)}"
                    f"| Loss: {loss.item():.4f}"
                )
    return model

def test_model():
    test_dataset = get_data_set(False)
    print("test_data_set length: " + str(len(test_dataset)))

    model = Netork()
    model.load_state_dict(torch.load("mnist.pth", weights_only=True))

    right = 0
    for i, (x, y) in enumerate(test_dataset):
        output = model(x)
        predict = output.argmax(1).item()
        if predict == y:
            right += 1
        else:
            # img_path = test_dataset.samples[i][0]
            img_path = "img_path"
            print(f"wrong case: predict = {predict} y = {y} img_path = {img_path}")
    
    sample_num = len(test_dataset)
    acc = right * 1.0 / sample_num
    print("test accuracy = %d / %d = %.31f" % (right, sample_num, acc))


if __name__ == "__main__":

    # model = train_model()

    # torch.save(model.state_dict(), "mnist.pth")

    test_model()