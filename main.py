import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from plmodels import TabularClassifier, ImageClassifier, ConvAutoEncoder  # Assuming the provided file is saved as `my_module.py`
from GenericDataset import *
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

TRAIN_DATA_FOLDER = "path_to_your_Dataset"
def main():
    # Example 1: Tabular Classifier
    print("Starting Tabular Classification...")

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train TabularClassifier
    tabular_model = TabularClassifier(RandomForestClassifier())
    tabular_model.fit(X_train, y_train)
    predictions = tabular_model.predict(X_test)
    print(f"Tabular Classifier Confusion Matrix:\n{tabular_model.ConfusionMatrix(predictions, y_test)}")

    # Example 2: Image Classifier
    print("Starting Image Classification...")

    # Create synthetic image dataset
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    custom_data = GenericDatasetLoader(dataset_name=dataset_file, root_dir = TRAIN_DATA_FOLDER, batch_size=32)

    train_loader = custom_data.create_dataloader(split='train', transform=transform, shuffle=True)

    test_loader = custom_data.create_dataloader(split='test', transform=transform, shuffle=False)

    val_loader = custom_data.create_dataloader(split='val', transform=transform, shuffle = False)
    
    # Initialize and train ImageClassifier
    image_model = ImageClassifier(model_name="resnet18", num_classes=10, max_epochs=3)
    image_model.fit(train_dataloader=train_loader)
    print("Image Classifier Training Complete.")


if __name__ == "__main__":
    main()
