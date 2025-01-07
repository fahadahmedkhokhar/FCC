import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from datasets import GenericDatasetLoader  # Assuming you have this module available

# Define a function to load the dataset
def read_image_dataset(dataset_file):
    """
    Load dataset and return the dataloader for training data.
    :param dataset_file: Root directory of the dataset.
    :return: DataLoader for training images.
    """
    transform = transforms.Compose([
        transforms.Resize((320, 320)),  # Resize images to 320x320
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize images
    ])

    # Load dataset using GenericDatasetLoader
    custom_data = GenericDatasetLoader(dataset_name="CUSTOM", root_dir=dataset_file, batch_size=32)
    train_loader = custom_data.create_dataloader(split='train', transform=transform, shuffle=True)

    return train_loader

# Define FGSM Attack
def fgsm_attack(image, epsilon, gradient):
    """
    Apply FGSM attack on the input image.
    :param image: Original image tensor.
    :param epsilon: Perturbation amount.
    :param gradient: Gradient of the loss w.r.t. the input image.
    :return: Adversarial image tensor and perturbation.
    """
    sign_gradient = gradient.sign()
    perturbed_image = image + epsilon * sign_gradient
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Keep pixel values in [0, 1]
    perturbation = perturbed_image - image  # Calculate perturbation
    return perturbed_image, perturbation

# Define a simple model for testing
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 320 * 320, 10)  # Adjusted for 320x320 images

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Apply FGSM and save adversarial images and perturbations
def apply_fgsm_and_save(dataloader, model, epsilon, save_folder, device):
    """
    Apply FGSM attack to all images in the DataLoader and save them.
    :param dataloader: DataLoader for the dataset.
    :param model: Model to attack.
    :param epsilon: Perturbation amount for FGSM.
    :param save_folder: Folder to save adversarial images and perturbations.
    :param device: Device (CPU or GPU).
    """
    os.makedirs(save_folder, exist_ok=True)
    perturbation_folder = os.path.join(save_folder, "perturbations")
    adversarial_folder = os.path.join(save_folder, "adversarial_images")
    os.makedirs(perturbation_folder, exist_ok=True)
    os.makedirs(adversarial_folder, exist_ok=True)

    model.eval()  # Set model to evaluation mode
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Get gradient and create adversarial image
        gradient = images.grad.data
        adv_images, perturbations = fgsm_attack(images, epsilon, gradient)

        # Save adversarial images and perturbations
        for i in range(adv_images.size(0)):
            adv_path = os.path.join(adversarial_folder, f"batch_{batch_idx}_img_{i}.png")
            pert_path = os.path.join(perturbation_folder, f"batch_{batch_idx}_pert_{i}.png")

            save_image(adv_images[i], adv_path)
            save_image((perturbations[i] + 1) / 2, pert_path)  # Rescale perturbations to [0, 1]

        print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

# Main function
def main():
    # Parameters
    dataset_file = "/home/fahadk/Project/FCC/dataset/flower/anomaly/normal_30"
    save_folder = "./adversarial_images_with_perturbations"  # Folder to save adversarial images and perturbations
    epsilon = 0.2  # Perturbation strength
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataloader = read_image_dataset(dataset_file)

    # Initialize model and load to device
    model = SimpleCNN().to(device)

    # Apply FGSM attack and save images
    apply_fgsm_and_save(dataloader, model, epsilon, save_folder, device)

if __name__ == "__main__":
    main()
