from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import yaml

config = yaml.safe_load(open("config.yml"))

# Define a transform to normalize the data
transform = transforms.Compose(
    [transforms.Resize((512,512)),  # resize images to 512x512
     transforms.ToTensor(),  # convert image to PyTorch tensor
     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalize image data
     std=[0.229, 0.224, 0.225])])

# Load images from the "images" directory
dataset = datasets.ImageFolder('images', transform=transform)

# Train and test split
train_size = int(len(dataset) * config["TRAIN_SPLIT"])
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create a data loader
train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)
