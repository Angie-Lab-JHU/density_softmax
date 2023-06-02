import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class MNISTDataloader(Dataset):
    def __init__(self, path, sample_paths, class_labels):
        self.image_transformer = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        self.path = path
        self.sample_paths, self.class_labels = sample_paths, class_labels

    def get_image(self, sample_path):
        img = Image.open(sample_path)
        return self.image_transformer(img)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        sample = self.get_image(self.path + self.sample_paths[index])
        class_label = self.class_labels[index]
        return sample, class_label


class MNIST_Test_Dataloader(MNISTDataloader):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self.image_transformer = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
