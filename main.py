import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import pupil_net
import torch.nn as nn
import torch.optim as optim

class PupilDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for subdir in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, subdir)):
                avi_file = os.path.join(self.root_dir, subdir, subdir + '.avi')
                txt_file = os.path.join(self.root_dir, subdir, subdir + '.txt')
                if os.path.exists(avi_file) and os.path.exists(txt_file):
                    samples.append((avi_file, txt_file))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        avi_file, txt_file = self.samples[idx]
        cap = cv2.VideoCapture(avi_file)
        frames = []
        annotations = []

        frame_skip = 10  # Skip every X frames
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame % frame_skip == 0:
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            current_frame += 1
        

        cap.release()
        
        with open(txt_file, 'r') as f:
            for line in f:
                x, y = line.strip().split()
                annotations.append((float(x), float(y)))

        # Pair each frame with its corresponding annotation
        return [(frame, torch.FloatTensor(annotation)) for frame, annotation in zip(frames, annotations)]


# Define a transform for preprocessing the frames
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create an instance of our dataset
pupil_dataset = PupilDataset(root_dir='LPW', transform=transform)

# Use DataLoader to handle batching
data_loader = DataLoader(pupil_dataset, batch_size=1, shuffle=True)

# Create an instance of our PupilNet
model = pupil_net.PupilNet()

# Loss function
criterion = nn.MSELoss()

# Optimizer (example: using Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for i, frame_label_pairs in enumerate(data_loader, 0):
        for j, (frame, label) in enumerate(frame_label_pairs):
            # frame = frame.squeeze(0)  # Remove unnecessary dimensions, if any
            # label = label.squeeze(0)  # Same for the label

            # Forward + backward + optimize
            optimizer.zero_grad()
            output = model(frame)  # frame should have shape [1, 3, 224, 224]
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # Print statistics
            # if j % 100 == 99:    # Print every 100 mini-batches
            #     print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Frame: {j + 1}, Loss: {loss.item()}')

            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Frame: {j + 1}, Loss: {loss.item()}')

print('Finished Training')

torch.save(model.state_dict(), 'pupil_net.pth')

# Load the trained model
model = pupil_net.PupilNet()
model.load_state_dict(torch.load('pupil_net.pth'))
model.eval()  # Set the model to evaluation mode

