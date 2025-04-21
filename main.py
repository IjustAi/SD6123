import time
import io
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# --- 配置设备 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 图像预处理 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# --- 自定义Dataset类 ---
class ISIC2019ParquetDataset(Dataset):
    def __init__(self, parquet_file, transform=None):
        self.data = pd.read_parquet(parquet_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        image_field = sample['image']
        image_bytes = image_field['bytes']
        label = sample['label']

        if isinstance(image_bytes, bytes):
            image = Image.open(io.BytesIO(image_bytes))
        else:
            raise ValueError(f"Unsupported image format: {type(image_bytes)}")

        if self.transform:
            image = self.transform(image)

        return image, label


# --- 主程序 ---
if __name__ == "__main__":

    # --- 加载数据 ---
    train_dataset = ISIC2019ParquetDataset('train-00000-of-00001.parquet', transform=transform)
    test_dataset = ISIC2019ParquetDataset('test-00000-of-00001.parquet', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # --- 定义模型 ---
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 8)  # 输出8类
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    total_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        start_time = time.time()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            running_total += labels.size(0)
            running_correct += preds.eq(labels).sum().item()

            # --- 实时输出每100个batch的训练状态 ---
            if (batch_idx + 1) % 1 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = running_loss / (batch_idx + 1)
                acc = 100. * running_correct / running_total
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

        # --- 每个epoch结束后做验证 ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, preds = outputs.max(1)
                test_total += labels.size(0)
                test_correct += preds.eq(labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * running_correct / running_total
        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"--- Epoch [{epoch}/{num_epochs}] Completed in {epoch_time:.2f}s ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test  Loss: {avg_test_loss:.4f} | Test  Acc: {test_acc:.2f}%\n")

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"Training Completed in {total_training_time / 60:.2f} minutes.")

    # --- 保存模型 ---
    torch.save(model.state_dict(), "resnet18_fedisic2019.pth")
    print("Model saved as resnet18_fedisic2019.pth")

    # --- 绘制训练曲线 ---
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()