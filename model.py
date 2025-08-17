import torch
import torch.nn as nn
import torchvision.models as models

class AnimalSpeciesClassifier(nn.Module):
    def __init__(self, num_classes=40):
        super(AnimalSpeciesClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze all layers except the final layer
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Replace the final fully connected layer
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
        """Unfreeze the last few layers for fine-tuning"""
        # Get all children of the model
        children = list(self.resnet.children())
        
        # Unfreeze the last num_layers
        for child in children[-num_layers:]:
            for param in child.parameters():
                param.requires_grad = True
                
    def save_model(self, path):
        """Save the model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.resnet.fc[-1].out_features
        }, path)
        
    @classmethod
    def load_model(cls, path):
        """Load a saved model"""
        checkpoint = torch.load(path)
        model = cls(num_classes=checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 