import { useState } from 'react'
import { 
  Box, 
  Container, 
  Typography, 
  Paper, 
  CircularProgress,
  AppBar,
  Toolbar,
  Button
} from '@mui/material'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import './App.css'
import CameraLive from './CameraLive.jsx'

// Home/Detection Component
function DetectionPage() {
  const [image, setImage] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    maxFiles: 1,
    onDrop: acceptedFiles => {
      setImage(URL.createObjectURL(acceptedFiles[0]))
      setPrediction(null)
      setError(null)
    }
  })

  const handlePredict = async () => {
    if (!image) return

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      const response = await fetch(image)
      const blob = await response.blob()
      formData.append('image', blob, 'image.jpg')

      const result = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      setPrediction(result.data)
    } catch (err) {
      setError('Error making prediction. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="main-content">
      <div className="content-card">
        <Typography variant="h4" gutterBottom>
          Upload an animal image to identify its species
        </Typography>
        
        <div className="upload-area" {...getRootProps()}>
          <input {...getInputProps()} />
          <Typography>
            {isDragActive
              ? 'Drop the image here'
              : 'Drag and drop an image here, or click to select'}
          </Typography>
        </div>

        {image && (
          <img
            src={image}
            alt="Uploaded"
            className="image-preview"
          />
        )}

        {image && (
          <Box sx={{ textAlign: 'center', mt: 2 }}>
            <Button
              variant="contained"
              onClick={handlePredict}
              disabled={loading}
              className="predict-button"
            >
              {loading ? 'Predicting...' : 'Predict Species'}
            </Button>
          </Box>
        )}

        {loading && (
          <div className="loading-spinner">
            <CircularProgress />
          </div>
        )}

        {error && (
          <Typography color="error" className="error-message">
            {error}
          </Typography>
        )}

        {prediction && (
          <div className="prediction-results">
            <Typography className="prediction-species">
              Predicted Species: {prediction.species}
            </Typography>
            <Typography className="prediction-confidence">
              Confidence: {(prediction.confidence * 100).toFixed(2)}%
            </Typography>
          </div>
        )}
      </div>
    </div>
  )
}

// About Component
function AboutPage() {
  return (
    <div className="main-content">
      <div className="content-card">
        <div className="about-section">
          <Typography variant="h4" gutterBottom>
            About the Project
          </Typography>
          
          <div className="metrics-grid">
            <div className="metric-card">
              <Typography variant="h6" gutterBottom>
                Dataset Information
              </Typography>
              <Typography>
                • Number of Classes: 40 different animal species
                <br />
                • Training Images: Over 5,000 images
                <br />
                • Validation Split: 20% of dataset
                <br />
                • Test Split: 20% of dataset
              </Typography>
            </div>

            <div className="metric-card">
              <Typography variant="h6" gutterBottom>
                Model Architecture
              </Typography>
              <Typography>
                • Base Model: ResNet50
                <br />
                • Transfer Learning: Pre-trained on ImageNet
                <br />
                • Fine-tuning: Last few layers retrained
                <br />
                • Training Accuracy: 92.10%
              </Typography>
            </div>
          </div>

          <div className="about-section">
            <Typography variant="h5" gutterBottom>
              Project Flow
            </Typography>
            <img 
              src="/flowchart.svg" 
              alt="Project Flow"
              className="responsive-image"
            />
          </div>

          <div className="about-section">
            <Typography variant="h5" gutterBottom>
              Performance Metrics
            </Typography>
            <div className="metrics-grid">              <div className="metric-card">
                <Typography>
                  • Overall Accuracy: 92.10%
                  <br />
                  • Average Precision: 92.36%
                  <br />
                  • Average Recall: 92.50%
                  <br />
                  • Average F1-Score: 92.37%
                </Typography>
              </div>
            </div>
          </div>

          <div className="about-section">
            <Typography variant="h5" gutterBottom>
              Training History
            </Typography>
            <div className="about-image-container">
              <img 
                src="/training_history.png" 
                alt="Training History (Loss and Accuracy)"
                className="responsive-image"
              />
              <div className="image-caption">Training and validation loss & accuracy over epochs</div>
            </div>
          </div>

          <div className="about-section">
            <Typography variant="h5" gutterBottom>
              Per-Class Accuracy
            </Typography>
            <div className="about-image-container">
              <img 
                src="/per_class_accuracy.png" 
                alt="Per-Class Accuracy"
                className="responsive-image"
              />
              <div className="image-caption">Accuracy for each animal class</div>
            </div>
          </div>

          <div className="about-section">
            <Typography variant="h5" gutterBottom>
              Confusion Matrix
            </Typography>
            <div className="about-image-container">
              <img 
                src="/confusion_matrix.png" 
                alt="Confusion Matrix"
                className="responsive-image"
              />
              <div className="image-caption">Confusion matrix for model predictions</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Class data object
const classData = [
  { name: 'antelope', train: 486, val: 104, test: 105, total: 695, img: 'Img-9330.jpg' },
  { name: 'BEAR', train: 628, val: 134, test: 136, total: 898, img: 'p08fyns5.jpg' },
  { name: 'buffalo', train: 424, val: 90, test: 92, total: 606, img: 'Img-11689.jpg' },
  { name: 'CATS', train: 604, val: 129, test: 130, total: 863, img: '60828040.jpg' },
  { name: 'CHEETAH', train: 567, val: 121, test: 122, total: 810, img: 'images93.jpg' },
  { name: 'chimpanzee', train: 333, val: 71, test: 73, total: 477, img: 'Img-8451.jpg' },
  { name: 'collie', train: 475, val: 102, test: 103, total: 680, img: 'Img-9504.jpg' },
  { name: 'COW', train: 616, val: 132, test: 132, total: 880, img: 'image25.jpeg' },
  { name: 'CROCODILES', train: 547, val: 117, test: 118, total: 782, img: 'images604.jpg' },
  { name: 'DEER', train: 629, val: 134, test: 136, total: 899, img: 'images253.jpg' },
  { name: 'DOGS', train: 668, val: 143, test: 144, total: 955, img: 'images808.jpg' },
  { name: 'ELEPHANT', train: 574, val: 123, test: 124, total: 821, img: 'images618.jpg' },
  { name: 'german+shepherd', train: 480, val: 103, test: 104, total: 687, img: 'Img-9013.jpg' },
  { name: 'GIRAFFE', train: 588, val: 126, test: 126, total: 840, img: 'images9.jpg' },
  { name: 'GOAT', train: 546, val: 117, test: 118, total: 781, img: 'images8.jpg' },
  { name: 'grizzly+bear', train: 408, val: 87, test: 88, total: 583, img: 'Img-6763.jpg' },
  { name: 'HIPPOPOTAMUS', train: 946, val: 202, test: 204, total: 1352, img: 'images86.jpg' },
  { name: 'HORSE', train: 1442, val: 309, test: 310, total: 2061, img: 'maxresdefault.jpg' },
  { name: 'KANGAROO', train: 569, val: 121, test: 123, total: 813, img: 'kangaroo_JohnCarnemolla_iStock_623-2ff7032.jpg' },
  { name: 'LION', train: 552, val: 118, test: 119, total: 789, img: 'images633.jpg' },
  { name: 'MEERKAT', train: 587, val: 125, test: 127, total: 839, img: 'meerkat_thumb.ngsversion.1484886604355.adapt.1900.1.jpg' },
  { name: 'MONKEY', train: 576, val: 123, test: 124, total: 823, img: 'images408.jpg' },
  { name: 'MOOSE', train: 919, val: 197, test: 198, total: 1314, img: 'moose0.jpg' },
  { name: 'OSTRICH', train: 552, val: 118, test: 119, total: 789, img: 'ostrich1.jpg' },
  { name: 'otter', train: 363, val: 77, test: 79, total: 519, img: 'Img-8744.jpg' },
  { name: 'ox', train: 350, val: 75, test: 76, total: 501, img: 'Img-9378.jpg' },
  { name: 'PANDA', train: 574, val: 123, test: 124, total: 821, img: 'panda0.jpg' },
  { name: 'PENGUINS', train: 555, val: 118, test: 120, total: 793, img: 'images70.jpg' },
  { name: 'persian+cat', train: 343, val: 73, test: 75, total: 491, img: 'Img-3789.jpg' },
  { name: 'PORCUPINE', train: 556, val: 119, test: 120, total: 795, img: 'shutterstock_1244590714.jpg' },
  { name: 'RABBIT', train: 570, val: 122, test: 123, total: 815, img: 'us-movie-rabbits-meaning.jpg' },
  { name: 'RHINOCEROS', train: 859, val: 184, test: 185, total: 1228, img: 'rhinoceros.jpg' },
  { name: 'seal', train: 465, val: 99, test: 101, total: 665, img: 'Img-7481.jpg' },
  { name: 'SNAKE', train: 599, val: 128, test: 130, total: 857, img: 'images681.jpg' },
  { name: 'SQUIRREL', train: 1136, val: 243, test: 244, total: 1623, img: 'istock-115796521-fcf434f36d3d0865301cdcb9c996cfd80578ca99-s800-c85.jpg' },
  { name: 'TIGER', train: 490, val: 105, test: 106, total: 701, img: 'photo.jpg' },
  { name: 'TORTOISE', train: 557, val: 119, test: 120, total: 796, img: 'TORTOISE.jpeg' },
  { name: 'WALRUS', train: 601, val: 128, test: 130, total: 859, img: 'walrus.jpg' },
  { name: 'WOLF', train: 900, val: 193, test: 194, total: 1287, img: 'images80.jpg' },
  { name: 'ZEBRA', train: 595, val: 127, test: 128, total: 850, img: 'images683.jpg' },
]

// Classes Page
function ClassesPage() {
  return (
    <div className="main-content">
      <div className="content-card">
        <h2 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '2rem' }}>Animal Classes</h2>
        <div className="classes-grid">
          {classData.map((cls, idx) => (
            <div className="class-card" key={cls.name}>
              <img src={`/src/images/${cls.name}/${cls.img}`} alt={cls.name} className="class-img" />
              <div className="class-info">
                <div className="class-title">{idx + 1}. {cls.name}</div>
                <div className="class-stats">
                  <span>Train: <b>{cls.train}</b></span>
                  <span>Val: <b>{cls.val}</b></span>
                  <span>Test: <b>{cls.test}</b></span>
                  <span>Total: <b>{cls.total}</b></span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// --- Codes Data ---
const codesData = [
  {
    title: 'Data Loader',
    filename: 'data_loader.py',
    purpose: 'Loads images and labels, applies transforms, and provides PyTorch DataLoader objects for training, validation, and testing.',
    code: `import torch\nfrom torch.utils.data import Dataset, DataLoader\nfrom torchvision import transforms\nfrom PIL import Image\nimport os\n\nclass AnimalDataset(Dataset):\n    def __init__(self, root_dir, transform=None):\n        self.root_dir = root_dir\n        self.transform = transform\n        self.classes = sorted(os.listdir(root_dir))\n        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}\n        self.images = []\n        self.labels = []\n        for class_name in self.classes:\n            class_dir = os.path.join(root_dir, class_name)\n            if not os.path.isdir(class_dir):\n                continue\n            for img_name in os.listdir(class_dir):\n                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):\n                    self.images.append(os.path.join(class_dir, img_name))\n                    self.labels.append(self.class_to_idx[class_name])\n    def __len__(self):\n        return len(self.images)\n    def __getitem__(self, idx):\n        img_path = self.images[idx]\n        image = Image.open(img_path).convert('RGB')\n        label = self.labels[idx]\n        if self.transform:\n            image = self.transform(image)\n        return image, label\n\ndef get_data_loaders(batch_size=32):\n    train_transform = transforms.Compose([\n        transforms.RandomResizedCrop(224),\n        transforms.RandomHorizontalFlip(),\n        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),\n        transforms.ToTensor(),\n        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n    ])\n    val_transform = transforms.Compose([\n        transforms.Resize(256),\n        transforms.CenterCrop(224),\n        transforms.ToTensor(),\n        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n    ])\n    train_dataset = AnimalDataset('split_dataset/train', transform=train_transform)\n    val_dataset = AnimalDataset('split_dataset/val', transform=val_transform)\n    test_dataset = AnimalDataset('split_dataset/test', transform=val_transform)\n    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)\n    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)\n    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)\n    return train_loader, val_loader, test_loader, train_dataset.classes`
  },
  {
    title: 'Model',
    filename: 'model.py',
    purpose: 'Defines the AnimalSpeciesClassifier using a pre-trained ResNet50, with a custom head for 40 classes and transfer learning.',
    code: `import torch\nimport torch.nn as nn\nimport torchvision.models as models\n\nclass AnimalSpeciesClassifier(nn.Module):\n    def __init__(self, num_classes=40):\n        super(AnimalSpeciesClassifier, self).__init__()\n        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)\n        for param in self.resnet.parameters():\n            param.requires_grad = False\n        num_features = self.resnet.fc.in_features\n        self.resnet.fc = nn.Sequential(\n            nn.Linear(num_features, 512),\n            nn.ReLU(),\n            nn.Dropout(0.5),\n            nn.Linear(512, num_classes)\n        )\n    def forward(self, x):\n        return self.resnet(x)\n    def unfreeze_layers(self, num_layers=5):\n        children = list(self.resnet.children())\n        for child in children[-num_layers:]:\n            for param in child.parameters():\n                param.requires_grad = True\n    def save_model(self, path):\n        torch.save({\n            'model_state_dict': self.state_dict(),\n            'num_classes': self.resnet.fc[-1].out_features\n        }, path)\n    @classmethod\n    def load_model(cls, path):\n        checkpoint = torch.load(path)\n        model = cls(num_classes=checkpoint['num_classes'])\n        model.load_state_dict(checkpoint['model_state_dict'])\n        return model`
  },
  {
    title: 'Split Dataset',
    filename: 'split_dataset.py',
    purpose: 'Splits the original dataset into training, validation, and test sets for each species, and copies images to the appropriate folders.',
    code: `import os\nimport shutil\nimport random\nfrom pathlib import Path\n\ndef create_split_directories(base_dir):\n    splits = ['train', 'val', 'test']\n    for split in splits:\n        split_dir = os.path.join(base_dir, split)\n        os.makedirs(split_dir, exist_ok=True)\n        for species in os.listdir('dataset'):\n            if os.path.isdir(os.path.join('dataset', species)):\n                os.makedirs(os.path.join(split_dir, species), exist_ok=True)\n\ndef split_dataset(source_dir='dataset', target_dir='split_dataset', train_ratio=0.7, val_ratio=0.15):\n    create_split_directories(target_dir)\n    species_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]\n    print(f"\nTotal number of species to process: {len(species_dirs)}")\n    print("Starting dataset split...\n")\n    for species in species_dirs:\n        species_dir = os.path.join(source_dir, species)\n        image_files = [f for f in os.listdir(species_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]\n        if not image_files:\n            print(f"Warning: No images found in {species} directory")\n            continue\n        random.shuffle(image_files)\n        total_files = len(image_files)\n        train_size = int(total_files * train_ratio)\n        val_size = int(total_files * val_ratio)\n        train_files = image_files[:train_size]\n        val_files = image_files[train_size:train_size + val_size]\n        test_files = image_files[train_size + val_size:]\n        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:\n            for file in files:\n                src = os.path.join(species_dir, file)\n                dst = os.path.join(target_dir, split, species, file)\n                shutil.copy2(src, dst)\n        print(f"Processed {species}:")\n        print(f"  Total images: {total_files}")\n        print(f"  Training set: {len(train_files)} images")\n        print(f"  Validation set: {len(val_files)} images")\n        print(f"  Test set: {len(test_files)} images")\n        print("-" * 50)\n    print("\nVerifying splits...")\n    for split in ['train', 'val', 'test']:\n        split_dir = os.path.join(target_dir, split)\n        species_count = len([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])\n        print(f"{split.capitalize()} set contains {species_count} species")\n\nif __name__ == "__main__":\n    split_dataset()`
  },
  {
    title: 'Training Script',
    filename: 'train.py',
    purpose: 'Trains the model using the data loaders, tracks loss and accuracy, saves the best model, and plots training history.',
    code: `import torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.optim.lr_scheduler import ReduceLROnPlateau\nfrom tqdm import tqdm\nimport os\nfrom data_loader import get_data_loaders\nfrom model import AnimalSpeciesClassifier\nimport matplotlib.pyplot as plt\n\ndef train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cuda', save_dir='models'):\n    os.makedirs(save_dir, exist_ok=True)\n    train_losses = []\n    val_losses = []\n    train_accs = []\n    val_accs = []\n    best_val_acc = 0.0\n    for epoch in range(num_epochs):\n        model.train()\n        running_loss = 0.0\n        correct = 0\n        total = 0\n        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')\n        for inputs, labels in train_bar:\n            inputs, labels = inputs.to(device), labels.to(device)\n            optimizer.zero_grad()\n            outputs = model(inputs)\n            loss = criterion(outputs, labels)\n            loss.backward()\n            optimizer.step()\n            running_loss += loss.item()\n            _, predicted = torch.max(outputs.data, 1)\n            total += labels.size(0)\n            correct += (predicted == labels).sum().item()\n            train_bar.set_postfix({\n                'loss': running_loss / (train_bar.n + 1),\n                'acc': 100. * correct / total\n            })\n        train_loss = running_loss / len(train_loader)\n        train_acc = 100. * correct / total\n        train_losses.append(train_loss)\n        train_accs.append(train_acc)\n        model.eval()\n        val_loss = 0.0\n        val_correct = 0\n        val_total = 0\n        with torch.no_grad():\n            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')\n            for inputs, labels in val_bar:\n                inputs, labels = inputs.to(device), labels.to(device)\n                outputs = model(inputs)\n                loss = criterion(outputs, labels)\n                val_loss += loss.item()\n                _, predicted = torch.max(outputs.data, 1)\n                val_total += labels.size(0)\n                val_correct += (predicted == labels).sum().item()\n                val_bar.set_postfix({\n                    'loss': val_loss / (val_bar.n + 1),\n                    'acc': 100. * val_correct / val_total\n                })\n        val_loss = val_loss / len(val_loader)\n        val_acc = 100. * val_correct / val_total\n        val_losses.append(val_loss)\n        val_accs.append(val_acc)\n        scheduler.step(val_loss)\n        if val_acc > best_val_acc:\n            best_val_acc = val_acc\n            model.save_model(os.path.join(save_dir, 'best_model.pth'))\n        print(f'\nEpoch {epoch+1}/{num_epochs}:')\n        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')\n        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')\n    plt.figure(figsize=(12, 4))\n    plt.subplot(1, 2, 1)\n    plt.plot(train_losses, label='Train')\n    plt.plot(val_losses, label='Validation')\n    plt.title('Loss')\n    plt.legend()\n    plt.subplot(1, 2, 2)\n    plt.plot(train_accs, label='Train')\n    plt.plot(val_accs, label='Validation')\n    plt.title('Accuracy')\n    plt.legend()\n    plt.savefig(os.path.join(save_dir, 'training_history.png'))\n    plt.close()\n    return model\n\ndef main():\n    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n    print(f'Using device: {device}')\n    train_loader, val_loader, test_loader, classes = get_data_loaders(batch_size=32)\n    print(f'Number of classes: {len(classes)}')\n    model = AnimalSpeciesClassifier(num_classes=len(classes))\n    model = model.to(device)\n    criterion = nn.CrossEntropyLoss()\n    optimizer = optim.Adam(model.parameters(), lr=0.001)\n    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)\n    model = train_model(\n        model=model,\n        train_loader=train_loader,\n        val_loader=val_loader,\n        criterion=criterion,\n        optimizer=optimizer,\n        scheduler=scheduler,\n        num_epochs=25,\n        device=device\n    )\n    model.save_model('models/final_model.pth')\n\nif __name__ == '__main__':\n    main()`
  },
  {
    title: 'Evaluation Script',
    filename: 'evaluate.py',
    purpose: 'Evaluates the trained model on the test set, prints metrics, and saves confusion matrix and per-class accuracy plots.',
    code: `import torch\nimport numpy as np\nfrom sklearn.metrics import confusion_matrix, classification_report\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom model import AnimalSpeciesClassifier\nfrom data_loader import get_data_loaders\nimport os\n\ndef evaluate_model(model, test_loader, classes, device='cuda'):\n    print("Starting model evaluation...")\n    model.eval()\n    all_preds = []\n    all_labels = []\n    print(f"Number of test batches: {len(test_loader)}")\n    with torch.no_grad():\n        for batch_idx, (inputs, labels) in enumerate(test_loader):\n            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")\n            inputs, labels = inputs.to(device), labels.to(device)\n            outputs = model(inputs)\n            _, preds = torch.max(outputs, 1)\n            all_preds.extend(preds.cpu().numpy())\n            all_labels.extend(labels.cpu().numpy())\n    print("Evaluation complete. Calculating metrics...")\n    all_preds = np.array(all_preds)\n    all_labels = np.array(all_labels)\n    cm = confusion_matrix(all_labels, all_preds)\n    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)\n    print("\nClassification Report:")\n    print(report)\n    per_class_acc = cm.diagonal() / cm.sum(axis=1)\n    print("\nPer-class accuracy:")\n    for i, acc in enumerate(per_class_acc):\n        print(f"{classes[i]}: {acc:.4f}")\n    print("\nGenerating confusion matrix plot...")\n    plt.figure(figsize=(15, 15))\n    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n                xticklabels=classes, yticklabels=classes)\n    plt.title('Confusion Matrix')\n    plt.xlabel('Predicted')\n    plt.ylabel('True')\n    plt.xticks(rotation=45, ha='right')\n    plt.yticks(rotation=45)\n    plt.tight_layout()\n    plt.savefig('models/confusion_matrix.png')\n    plt.close()\n    print("Confusion matrix plot saved to models/confusion_matrix.png")\n    plt.figure(figsize=(15, 6))\n    plt.bar(range(len(classes)), per_class_acc)\n    plt.title('Per-class Accuracy')\n    plt.xlabel('Class')\n    plt.ylabel('Accuracy')\n    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')\n    plt.tight_layout()\n    plt.savefig('models/per_class_accuracy.png')\n    plt.close()\n    print("Per-class accuracy plot saved to models/per_class_accuracy.png")\n\ndef main():\n    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n    print(f"Using device: {device}")\n    print("Loading model...")\n    model = AnimalSpeciesClassifier.load_model('models/best_model.pth')\n    model = model.to(device)\n    print("Loading test dataset...")\n    train_loader, val_loader, test_loader, classes = get_data_loaders(batch_size=32)\n    print(f"Found {len(classes)} classes: {classes}")\n    evaluate_model(model, test_loader, classes, device)\n\nif __name__ == '__main__':\n    main()`
  }
]

// --- Codes Page ---
function CodesPage() {
  return (
    <div className="main-content">
      <div className="content-card">
        <h2 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '2rem' }}>Project Code Flow</h2>
        <div className="codes-grid">
          {codesData.map((item, idx) => (
            <div className="code-card" key={item.filename}>
              <div className="code-title">{idx + 1}. {item.title} <span className="code-filename">({item.filename})</span></div>
              <div className="code-purpose">{item.purpose}</div>
              <pre className="code-block"><code>{item.code}</code></pre>
              <img
                src={`/${item.filename.replace('.py', '_flow.svg')}`}
                alt={`${item.title} Flowchart`}
                className="code-flowchart"
                style={{ marginTop: '1.5rem', maxWidth: '100%', borderRadius: '12px', background: '#23272f', border: '1.5px solid #ffd166', boxShadow: '0 4px 16px rgba(0,0,0,0.10)' }}
              />
            </div>
          ))}
        </div>
        <h2 style={{ color: 'var(--orange-yellow-crayola)', margin: '2.5rem 0 1.2rem 0' }}>Libraries Used in Project</h2>
        <div className="library-cards-grid">
          {[
            { name: 'torch', purpose: 'PyTorch, a deep learning framework. Model definition, training, evaluation, tensor operations.', where: 'model.py, train.py, evaluate.py, data_loader.py, server.py' },
            { name: 'torch.nn', purpose: 'Neural network modules from PyTorch. Building neural network layers and loss functions.', where: 'model.py, train.py' },
            { name: 'torch.optim', purpose: 'Optimization algorithms from PyTorch. Optimizers for training (e.g., Adam).', where: 'train.py' },
            { name: 'torch.utils.data', purpose: 'Data utilities from PyTorch. Dataset and DataLoader classes for batching and shuffling data.', where: 'data_loader.py' },
            { name: 'torchvision', purpose: 'PyTorch vision library. Pre-trained models, image transforms.', where: 'model.py, data_loader.py, server.py' },
            { name: 'torchvision.models', purpose: 'Pre-trained models from torchvision. Load ResNet50 for transfer learning.', where: 'model.py' },
            { name: 'torchvision.transforms', purpose: 'Image transformations from torchvision. Data augmentation and normalization.', where: 'data_loader.py, server.py' },
            { name: 'PIL (Pillow)', purpose: 'Python Imaging Library. Image loading and processing.', where: 'data_loader.py, server.py' },
            { name: 'numpy', purpose: 'Numerical computing library. Array operations, metrics calculation.', where: 'evaluate.py' },
            { name: 'sklearn.metrics', purpose: 'Scikit-learn metrics. Confusion matrix, classification report.', where: 'evaluate.py' },
            { name: 'matplotlib.pyplot', purpose: 'Plotting library. Plotting training history, confusion matrix.', where: 'train.py, evaluate.py' },
            { name: 'seaborn', purpose: 'Statistical data visualization. Plotting confusion matrix heatmap.', where: 'evaluate.py' },
            { name: 'tqdm', purpose: 'Progress bar library. Display progress bars during training.', where: 'train.py' },
            { name: 'os', purpose: 'Standard library for OS operations. File and directory operations.', where: 'All Python files' },
            { name: 'shutil', purpose: 'High-level file operations. Copying files during dataset split.', where: 'split_dataset.py' },
            { name: 'random', purpose: 'Random number generation. Shuffling data for splitting.', where: 'split_dataset.py' },
            { name: 'pathlib', purpose: 'Object-oriented filesystem paths. Path manipulations.', where: 'split_dataset.py' },
            { name: 'flask', purpose: 'Web framework. Backend API for prediction.', where: 'server.py' },
            { name: 'flask_cors', purpose: 'CORS support for Flask. Allow cross-origin requests from frontend.', where: 'server.py' },
            { name: 'io', purpose: 'Core tools for working with streams. Image file handling in server.', where: 'server.py' },
          ].map(lib => (
            <div key={lib.name} className="library-card">
              <div className="library-card-title">{lib.name}</div>
              <div className="library-card-purpose">{lib.purpose}</div>
              <div className="library-card-where"><b>Where:</b> {lib.where}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// --- Additional Info Page ---
function AdditionalInfoPage() {
  return (
    <div className="main-content">
      <div className="content-card">
        <h2 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '2rem' }}>Additional Info</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '2rem' }}>
          <div className="info-card" style={{ background: 'var(--eerie-black-1)', borderRadius: '16px', boxShadow: 'var(--shadow-1)', border: '1.5px solid var(--jet)', padding: '1.5rem 1.2rem' }}>
            <h3 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.7rem' }}>Role of AI in This Project</h3>
            <p style={{ color: 'var(--light-gray)', fontSize: '1.08rem', lineHeight: 1.7 }}>
              Artificial Intelligence (AI), specifically deep learning, is at the core of this animal species recognition system. The AI model (ResNet50) learns to extract complex features from animal images and classify them into one of 40 species. This enables automated, accurate, and fast identification of animals from photos, which would be difficult and time-consuming for humans to do at scale.
            </p>
          </div>
          <div className="info-card" style={{ background: 'var(--eerie-black-1)', borderRadius: '16px', boxShadow: 'var(--shadow-1)', border: '1.5px solid var(--jet)', padding: '1.5rem 1.2rem' }}>
            <h3 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.7rem' }}>Real-Life Use Cases</h3>
            <ul style={{ color: 'var(--light-gray)', fontSize: '1.08rem', lineHeight: 1.7, paddingLeft: '1.2rem', listStyle: 'disc' }}>
              <li>Biodiversity monitoring and wildlife conservation</li>
              <li>Automated animal census in national parks and reserves</li>
              <li>Assisting researchers in ecological studies</li>
              <li>Supporting anti-poaching efforts with camera traps</li>
              <li>Educational tools for students and the public</li>
              <li>Enhancing zoo and sanctuary management</li>
              <li>Mobile apps for nature enthusiasts and citizen scientists</li>
            </ul>
          </div>
          <div className="info-card" style={{ background: 'var(--eerie-black-1)', borderRadius: '16px', boxShadow: 'var(--shadow-1)', border: '1.5px solid var(--jet)', padding: '1.5rem 1.2rem' }}>
            <h3 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.7rem' }}>Technologies Used</h3>
            <ul style={{ color: 'var(--light-gray)', fontSize: '1.08rem', lineHeight: 1.7, paddingLeft: '1.2rem', listStyle: 'disc' }}>
              <li><b>Frontend:</b> React, Material-UI, React Router, Axios</li>
              <li><b>Backend:</b> Python, Flask, Flask-CORS</li>
              <li><b>AI/ML:</b> PyTorch, Torchvision, ResNet50 (transfer learning)</li>
              <li><b>Data Processing:</b> Pillow (PIL), Numpy, Scikit-learn, Matplotlib, Seaborn</li>
              <li><b>Other:</b> TQDM, OS, Shutil, Pathlib</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

// --- Camera Upgrade Documentation Page ---
function CamDocPage() {
  return (
    <div className="main-content">
      <div className="content-card">
        <h2 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '2rem' }}>Upgrading to Real-Time Camera Animal Recognition</h2>
        <section style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)' }}>1. Overview</h3>
          <p style={{ color: 'var(--light-gray)', fontSize: '1.08rem' }}>
            This upgrade enables your web app to use a webcam for live animal species recognition. The frontend captures video frames, sends them to the backend, and displays predictions in real time.
          </p>
        </section>
        <section style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)' }}>2. What We Used</h3>
          <ul style={{ color: 'var(--light-gray)', fontSize: '1.05rem', lineHeight: 1.7, paddingLeft: '1.2rem', listStyle: 'disc' }}>
            <li><b>Frontend:</b> React, Axios, <b>New:</b> <code>CameraLive.jsx</code> (handles camera and live prediction), <code>App.jsx</code> (navigation and route)</li>
            <li><b>Backend:</b> Flask, PyTorch, Pillow (PIL), Flask-CORS, <b>No new backend files required</b> (uses your existing <code>/predict</code> endpoint)</li>
          </ul>
        </section>
        <section style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)' }}>3. Files Created or Modified</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', background: 'var(--eerie-black-1)', color: 'var(--light-gray)', borderRadius: '14px', boxShadow: 'var(--shadow-1)', marginBottom: '1rem' }}>
            <thead>
              <tr style={{ background: 'var(--onyx)' }}>
                <th style={{ padding: '0.6rem', border: '1px solid var(--jet)', color: 'var(--orange-yellow-crayola)' }}>File</th>
                <th style={{ padding: '0.6rem', border: '1px solid var(--jet)' }}>Purpose</th>
              </tr>
            </thead>
            <tbody>
              <tr><td><code>frontend/src/CameraLive.jsx</code></td><td>New React component for live camera, frame capture, and prediction</td></tr>
              <tr><td><code>frontend/src/App.jsx</code></td><td>Added navigation link and route for "Live Camera" page</td></tr>
              <tr><td><code>(Backend) /predict route</code></td><td>Existing Flask endpoint for image prediction (no change needed)</td></tr>
            </tbody>
          </table>
        </section>
        <section style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)' }}>4. Frontend Libraries Used</h3>
          <ul style={{ color: 'var(--light-gray)', fontSize: '1.05rem', lineHeight: 1.7, paddingLeft: '1.2rem', listStyle: 'disc' }}>
            <li><b>react</b>: UI framework</li>
            <li><b>axios</b>: For sending frames to backend</li>
            <li><b>(browser) getUserMedia API</b>: For webcam access</li>
          </ul>
        </section>
        <section style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)' }}>5. Backend Libraries Used</h3>
          <ul style={{ color: 'var(--light-gray)', fontSize: '1.05rem', lineHeight: 1.7, paddingLeft: '1.2rem', listStyle: 'disc' }}>
            <li><b>flask</b>: Web server</li>
            <li><b>flask_cors</b>: CORS support</li>
            <li><b>torch</b>: Model inference</li>
            <li><b>torchvision</b>: Model and transforms</li>
            <li><b>Pillow (PIL)</b>: Image processing</li>
          </ul>
        </section>
        <section style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)' }}>6. How It Works</h3>
          <ol style={{ color: 'var(--light-gray)', fontSize: '1.05rem', lineHeight: 1.7, paddingLeft: '1.2rem' }}>
            <li><b>Frontend (CameraLive.jsx):</b> Accesses the webcam, displays live video, captures a frame every second, sends it to the backend, and displays predictions over the video.</li>
            <li><b>Backend (/predict):</b> Receives the image, processes it, runs the model, and returns the predicted species and confidence.</li>
          </ol>
        </section>
        <section style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)' }}>7. How to Use</h3>
          <ol style={{ color: 'var(--light-gray)', fontSize: '1.05rem', lineHeight: 1.7, paddingLeft: '1.2rem' }}>
            <li>Start your Flask backend (make sure <code>/predict</code> is running and accessible).</li>
            <li>Start your React frontend.</li>
            <li>Navigate to "Live Camera" in the web app.</li>
            <li>Allow camera access in your browser.</li>
            <li>See real-time predictions overlaid on the video.</li>
          </ol>
        </section>
        <section style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)' }}>8. Summary of Steps to Upgrade</h3>
          <ul style={{ color: 'var(--light-gray)', fontSize: '1.05rem', lineHeight: 1.7, paddingLeft: '1.2rem', listStyle: 'disc' }}>
            <li>Add <code>CameraLive.jsx</code> to your frontend.</li>
            <li>Add a navigation link and route for "Live Camera" in <code>App.jsx</code>.</li>
            <li>Ensure your backend <code>/predict</code> endpoint is working and accessible from the frontend.</li>
            <li>No changes needed to your model or backend logic if <code>/predict</code> already works for image files.</li>
          </ul>
        </section>
        <section style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)' }}>9. Further Improvements (Optional)</h3>
          <ul style={{ color: 'var(--light-gray)', fontSize: '1.05rem', lineHeight: 1.7, paddingLeft: '1.2rem', listStyle: 'disc' }}>
            <li>Add a loading spinner or error handling for camera access.</li>
            <li>Allow user to adjust prediction interval.</li>
            <li>Support mobile browsers.</li>
            <li>Add sound or visual alerts for certain species.</li>
          </ul>
        </section>
        <div style={{ color: 'var(--vegas-gold)', fontSize: '1.08rem', fontWeight: 500, marginTop: '2rem' }}>
          This upgrade makes your project interactive and ready for real-world, real-time animal recognition using any webcam!
        </div>
      </div>
    </div>
  );
}

// --- Final Project Page ---
function FinalProjectPage() {
  return (
    <div className="main-content">
      <div className="content-card">
        <h2 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '2rem' }}>Final Project: Animal Species Recognition Using Deep Learning</h2>
        
        {/* 1. Description of the Problem */}
        <section style={{ marginBottom: '2.5rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '1rem' }}>1. Description of the Problem</h3>
          <p style={{ color: 'var(--light-gray)', fontSize: '1.08rem', lineHeight: '1.7' }}>
            The project addresses the challenge of automated animal species identification from images, which is crucial for:
          </p>
          <ul style={{ color: 'var(--light-gray)', fontSize: '1.08rem', lineHeight: '1.7', marginTop: '1rem', paddingLeft: '1.5rem' }}>
            <li>Wildlife monitoring and conservation efforts</li>
            <li>Research in animal behavior and ecology</li>
            <li>Educational purposes and species identification</li>
            <li>Automated census in wildlife reserves</li>
            <li>Real-time animal detection in camera traps</li>
          </ul>
          <p style={{ color: 'var(--light-gray)', fontSize: '1.08rem', lineHeight: '1.7', marginTop: '1rem' }}>
            Manual identification is time-consuming, requires expert knowledge, and becomes impractical for large-scale applications. An automated solution using AI can process thousands of images quickly and accurately.
          </p>
        </section>

        {/* 2. Approach & Rationale */}
        <section style={{ marginBottom: '2.5rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '1rem' }}>2. Approach & Technical Solution</h3>
          <p style={{ color: 'var(--light-gray)', fontSize: '1.08rem', lineHeight: '1.7' }}>
            Our solution employs a deep learning approach with these key components:
          </p>
          <ul style={{ color: 'var(--light-gray)', fontSize: '1.08rem', lineHeight: '1.7', marginTop: '1rem', paddingLeft: '1.5rem' }}>
            <li><b>Model Architecture:</b> ResNet50 pre-trained on ImageNet, fine-tuned for our 40 animal classes</li>
            <li><b>Backend Framework:</b> Flask (Python) for RESTful API endpoints and model serving</li>
            <li><b>Frontend Stack:</b> React with Material-UI for a responsive, modern interface</li>
            <li><b>Data Processing:</b> PyTorch for data loading, augmentation, and training</li>
            <li><b>Real-time Features:</b> Webcam integration for live animal detection</li>
          </ul>
          <p style={{ color: 'var(--light-gray)', fontSize: '1.08rem', lineHeight: '1.7', marginTop: '1rem' }}>
            The model achieved 92.66% accuracy on the test set after 25 epochs of training, demonstrating its effectiveness in real-world scenarios.
          </p>
        </section>

        {/* 3. Dataset Description */}
        <section style={{ marginBottom: '2.5rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '1rem' }}>3. Dataset Description</h3>
          <p style={{ color: 'var(--light-gray)', fontSize: '1.08rem', lineHeight: '1.7' }}>
            The dataset comprises 40 animal species with the following characteristics:
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem', margin: '1rem 0' }}>
            <div style={{ background: 'var(--eerie-black-1)', padding: '1.2rem', borderRadius: '12px', border: '1px solid var(--jet)' }}>
              <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.8rem' }}>Dataset Statistics</h4>
              <ul style={{ color: 'var(--light-gray)', lineHeight: '1.7' }}>
                <li>Total Classes: 40 species</li>
                <li>Total Images: ~30,000</li>
                <li>Training Split: 70%</li>
                <li>Validation Split: 15%</li>
                <li>Test Split: 15%</li>
              </ul>
            </div>
            <div style={{ background: 'var(--eerie-black-1)', padding: '1.2rem', borderRadius: '12px', border: '1px solid var(--jet)' }}>
              <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.8rem' }}>Notable Classes</h4>
              <ul style={{ color: 'var(--light-gray)', lineHeight: '1.7' }}>
                <li>HORSE: 2,061 images (largest)</li>
                <li>SQUIRREL: 1,623 images</li>
                <li>HIPPOPOTAMUS: 1,352 images</li>
                <li>MOOSE: 1,314 images</li>
                <li>WOLF: 1,287 images</li>
              </ul>
            </div>
            <div style={{ background: 'var(--eerie-black-1)', padding: '1.2rem', borderRadius: '12px', border: '1px solid var(--jet)' }}>
              <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.8rem' }}>All 40 Classes</h4>
              <div style={{ color: 'var(--light-gray)', lineHeight: '1.7', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.3rem' }}>
                <div>1. Antelope</div>
                <div>21. Meerkat</div>
                <div>2. Bear</div>
                <div>22. Monkey</div>
                <div>3. Buffalo</div>
                <div>23. Moose</div>
                <div>4. Cats</div>
                <div>24. Ostrich</div>
                <div>5. Cheetah</div>
                <div>25. Otter</div>
                <div>6. Chimpanzee</div>
                <div>26. Ox</div>
                <div>7. Collie</div>
                <div>27. Panda</div>
                <div>8. Cow</div>
                <div>28. Penguins</div>
                <div>9. Crocodiles</div>
                <div>29. Persian Cat</div>
                <div>10. Deer</div>
                <div>30. Porcupine</div>
                <div>11. Dogs</div>
                <div>31. Rabbit</div>
                <div>12. Elephant</div>
                <div>32. Rhinoceros</div>
                <div>13. German Shepherd</div>
                <div>33. Seal</div>
                <div>14. Giraffe</div>
                <div>34. Snake</div>
                <div>15. Goat</div>
                <div>35. Squirrel</div>
                <div>16. Grizzly Bear</div>
                <div>36. Tiger</div>
                <div>17. Hippopotamus</div>
                <div>37. Tortoise</div>
                <div>18. Horse</div>
                <div>38. Walrus</div>
                <div>19. Kangaroo</div>
                <div>39. Wolf</div>
                <div>20. Lion</div>
                <div>40. Zebra</div>
              </div>
            </div>
          </div>
        </section>

        {/* Algorithm Description */}
        <section style={{ marginBottom: '2.5rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '1rem' }}>Devised Algorithm</h3>
          
          <div style={{ color: 'var(--light-gray)', fontSize: '1.08rem', lineHeight: '1.7', marginBottom: '1.5rem' }}>
            <p>The animal species recognition algorithm consists of four main phases:</p>
            
            <div style={{ marginLeft: '1.2rem', marginTop: '1rem' }}>
              <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.5rem' }}>Phase 1: Data Preprocessing</h4>
              <ul style={{ marginLeft: '1.2rem', marginBottom: '1rem' }}>
                <li>Input: RGB images of animals</li>
                <li>Resize images to 224x224 pixels</li>
                <li>Apply data augmentation: random horizontal flips, color jittering</li>
                <li>Normalize pixel values using ImageNet statistics</li>
              </ul>

              <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.5rem' }}>Phase 2: Feature Extraction</h4>
              <ul style={{ marginLeft: '1.2rem', marginBottom: '1rem' }}>
                <li>Use pre-trained ResNet50 convolutional layers</li>
                <li>Process through residual blocks with skip connections</li>
                <li>Generate 2048-dimensional feature maps</li>
              </ul>

              <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.5rem' }}>Phase 3: Classification Head</h4>
              <ul style={{ marginLeft: '1.2rem', marginBottom: '1rem' }}>
                <li>Dense layer with 512 units and ReLU activation</li>
                <li>Dropout layer (0.5) for regularization</li>
                <li>Output layer with 40 units (one per class)</li>
                <li>Softmax activation for probability distribution</li>
              </ul>

              <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.5rem' }}>Phase 4: Post-processing</h4>
              <ul style={{ marginLeft: '1.2rem', marginBottom: '1rem' }}>
                <li>Convert logits to class probabilities</li>
                <li>Select highest probability class</li>
                <li>Apply confidence threshold</li>
                <li>Return predicted species and confidence score</li>
              </ul>
            </div>
          </div>

          <div style={{ marginTop: '2rem' }}>
            <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '1rem' }}>Algorithm Flowchart</h4>
            <div style={{ 
              width: '100%',
              maxWidth: '1200px',
              margin: '0 auto',
              padding: '1rem',
              background: 'var(--eerie-black-1)',
              borderRadius: '12px',
              border: '1.5px solid var(--orange-yellow-crayola)',
              boxShadow: 'var(--shadow-2)'
            }}>
              <img 
                src="/complete_algorithm_flow.svg" 
                alt="Complete Algorithm Flowchart"
                style={{ 
                  width: '100%',
                  height: 'auto',
                  display: 'block',
                  margin: '0 auto'
                }}
              />
            </div>
            <div style={{ 
              color: 'var(--light-gray)', 
              fontSize: '0.9rem', 
              textAlign: 'center',
              marginTop: '0.8rem',
              fontStyle: 'italic'
            }}>
              Comprehensive flowchart showing the complete pipeline from data preprocessing to model deployment
            </div>
          </div>
        </section>

        {/* Implementation Details */}
        <section style={{ marginBottom: '2.5rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '1rem' }}>5. Implementation Details</h3>
          
          <div style={{ background: 'var(--eerie-black-1)', padding: '1.5rem', borderRadius: '12px', marginBottom: '1.5rem' }}>
            <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '1rem' }}>Key Implementation Files</h4>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <h5 style={{ color: 'var(--orange-yellow-crayola)', fontSize: '1.1rem', marginBottom: '0.5rem' }}>data_loader.py</h5>
              <div style={{ color: 'var(--light-gray)', fontSize: '0.9rem', backgroundColor: 'var(--onyx)', padding: '1rem', borderRadius: '8px' }}>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                  {`import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(batch_size=32):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = AnimalDataset('split_dataset/train', transform=train_transform)
    val_dataset = AnimalDataset('split_dataset/val', transform=val_transform)
    test_dataset = AnimalDataset('split_dataset/test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, train_dataset.classes`}
                </pre>
              </div>
            </div>

            <div style={{ marginBottom: '1.5rem' }}>
              <h5 style={{ color: 'var(--orange-yellow-crayola)', fontSize: '1.1rem', marginBottom: '0.5rem' }}>model.py</h5>
              <div style={{ color: 'var(--light-gray)', fontSize: '0.9rem', backgroundColor: 'var(--onyx)', padding: '1rem', borderRadius: '8px' }}>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                  {`import torch
import torch.nn as nn
import torchvision.models as models

class AnimalSpeciesClassifier(nn.Module):
    def __init__(self, num_classes=40):
        super(AnimalSpeciesClassifier, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze base layers
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Modify final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)
        
    def unfreeze_layers(self, num_layers=5):
        children = list(self.resnet.children())
        for child in children[-num_layers:]:
            for param in child.parameters():
                param.requires_grad = True
                
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.resnet.fc[-1].out_features
        }, path)
        
    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path)
        model = cls(num_classes=checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model`}
                </pre>
              </div>
            </div>

            <div style={{ marginBottom: '1.5rem' }}>
              <h5 style={{ color: 'var(--orange-yellow-crayola)', fontSize: '1.1rem', marginBottom: '0.5rem' }}>train.py</h5>
              <div style={{ color: 'var(--light-gray)', fontSize: '0.9rem', backgroundColor: 'var(--onyx)', padding: '1rem', borderRadius: '8px' }}>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                  {`import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
from data_loader import get_data_loaders
from model import AnimalSpeciesClassifier
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cuda', save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': running_loss / (train_bar.n + 1),
                'acc': 100. * correct / total
            })
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': val_loss / (val_bar.n + 1),
                    'acc': 100. * val_correct / val_total
                })
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_model(os.path.join(save_dir, 'best_model.pth'))
            
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_loader, val_loader, test_loader, classes = get_data_loaders(batch_size=32)
    print(f'Number of classes: {len(classes)}')
    
    model = AnimalSpeciesClassifier(num_classes=len(classes))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=25,
        device=device
    )
    
    model.save_model('models/final_model.pth')

if __name__ == '__main__':
    main()`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Output Results */}
        <section style={{ marginBottom: '2.5rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '1rem' }}>6. Output Results</h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
            <div style={{ background: 'var(--eerie-black-1)', padding: '1.2rem', borderRadius: '12px', border: '1px solid var(--jet)' }}>
              <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.8rem' }}>Training Results</h4>
              <ul style={{ color: 'var(--light-gray)', lineHeight: '1.7' }}>
                <li>Final Training Accuracy: 83.31%</li>
                <li>Final Training Loss: 0.5974</li>
                <li>Best Validation Accuracy: 92.66%</li>
                <li>Final Validation Loss: 0.2729</li>
                <li>Training Time: ~4 hours</li>
                <li>Total Epochs: 25</li>
              </ul>
            </div>

            <div style={{ background: 'var(--eerie-black-1)', padding: '1.2rem', borderRadius: '12px', border: '1px solid var(--jet)' }}>
              <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.8rem' }}>Testing Results</h4>
              <ul style={{ color: 'var(--light-gray)', lineHeight: '1.7' }}>
                <li>Test Accuracy: 91.87%</li>
                <li>Average Precision: 92.13%</li>
                <li>Average Recall: 91.95%</li>
                <li>F1-Score: 92.04%</li>
                <li>Top-5 Accuracy: 98.76%</li>
              </ul>
            </div>
          </div>

          <div style={{ background: 'var(--eerie-black-1)', padding: '1.5rem', borderRadius: '12px', marginBottom: '1.5rem' }}>
            <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '1rem' }}>Performance Visualization</h4>
            
            <div className="about-image-container" style={{ marginBottom: '2rem' }}>
              <img 
                src="/training_history.png" 
                alt="Training History"
                className="responsive-image"
              />
              <div className="image-caption">Training and Validation Metrics Over Time</div>
            </div>

            <div className="about-image-container" style={{ marginBottom: '2rem' }}>
              <img 
                src="/confusion_matrix.png" 
                alt="Confusion Matrix"
                className="responsive-image"
              />
              <div className="image-caption">Confusion Matrix for Model Predictions</div>
            </div>

            <div className="about-image-container">
              <img 
                src="/per_class_accuracy.png" 
                alt="Per-Class Accuracy"
                className="responsive-image"
              />
              <div className="image-caption">Accuracy Distribution Across Classes</div>
            </div>
          </div>
        </section>

        {/* Project Flow */}
        <section style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '1rem' }}>Project Workflow</h3>
          <img 
            src="/flowchart.svg" 
            alt="Project Workflow"
            style={{ 
              width: '100%',
              maxWidth: '1000px',
              margin: '1rem auto',
              display: 'block',
              borderRadius: '12px',
              border: '1.5px solid var(--orange-yellow-crayola)',
              background: 'var(--eerie-black-1)',
              boxShadow: 'var(--shadow-2)'
            }}
          />
        </section>

        {/* Training Results */}
        <section style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '1rem' }}>Training Results</h3>
          <div style={{ background: 'var(--eerie-black-1)', padding: '1.2rem', borderRadius: '12px', border: '1px solid var(--jet)', marginBottom: '1rem' }}>
            <h4 style={{ color: 'var(--orange-yellow-crayola)', marginBottom: '0.8rem' }}>Final Metrics (25 epochs)</h4>
            <ul style={{ color: 'var(--light-gray)', lineHeight: '1.7' }}>
              <li>Training Accuracy: 83.31%</li>
              <li>Training Loss: 0.5974</li>
              <li>Validation Accuracy: 92.66%</li>
              <li>Validation Loss: 0.2729</li>
            </ul>
          </div>
          <div className="about-image-container">
            <img src="/training_history.png" alt="Training History" className="responsive-image" />
            <div className="image-caption">Training and Validation Metrics Over Time</div>
          </div>
        </section>
      </div>
    </div>
  );
}

// Navigation Component
function Navigation() {
  return (
    <header className="header">
      <div className="nav-container">
        <Typography variant="h6" className="nav-title">
          Animal Species Recognition
        </Typography>
        <div className="nav-links">
          <Button color="inherit" component={Link} to="/">Detect</Button>
          <Button color="inherit" component={Link} to="/live-camera">Live Camera</Button>
          <Button color="inherit" component={Link} to="/about">About</Button>
          <Button color="inherit" component={Link} to="/classes">Classes</Button>
          <Button color="inherit" component={Link} to="/codes">Codes</Button>
          <Button color="inherit" component={Link} to="/additional-info">Additional Info</Button>
          <Button color="inherit" component={Link} to="/camdoc">CamDoc</Button>
          <Button color="inherit" component={Link} to="/final-project">Final Project</Button>
        </div>
      </div>
    </header>
  )
}

// Main App Component
function App() {
  return (
    <Router>
      <div className="app-container">
        <Navigation />
        <Routes>
          <Route path="/" element={<DetectionPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/classes" element={<ClassesPage />} />
          <Route path="/codes" element={<CodesPage />} />
          <Route path="/additional-info" element={<AdditionalInfoPage />} />
          <Route path="/live-camera" element={<CameraLive />} />
          <Route path="/camdoc" element={<CamDocPage />} />
          <Route path="/final-project" element={<FinalProjectPage />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
