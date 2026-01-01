import torch
import torch.nn as nn
import torchvision.models as models

class RetinaModel(nn.Module):
    def __init__(self, backbone_name='resnet101', num_classes=5):
        super(RetinaModel, self).__init__()
        if backbone_name == 'resnet101':
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        elif backbone_name == 'resnet152':
            self.backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class RetinaEnsemble(nn.Module):
    def __init__(self, num_classes=5):
        super(RetinaEnsemble, self).__init__()
        # Instantiate both models
        self.model_a = RetinaModel('resnet101', num_classes)
        self.model_b = RetinaModel('resnet152', num_classes)
        
    def forward(self, x):
        # Get logits from both models
        logits_a = self.model_a(x)
        logits_b = self.model_b(x)
        
        return logits_a, logits_b
    
    def get_ensemble_prediction(self, x):
        """
        Returns the soft-voting prediction (average probabilities).
        Useful for inference.
        """
        logits_a, logits_b = self.forward(x)
        probs_a = torch.softmax(logits_a, dim=1)
        probs_b = torch.softmax(logits_b, dim=1)
        
        avg_probs = (probs_a + probs_b) / 2.0
        return avg_probs

if __name__ == "__main__":
    # Test shapes
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = RetinaEnsemble(num_classes=5).to(device)
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    
    # Check forward pass
    la, lb = model(dummy_input)
    print(f"Logits A shape: {la.shape}")
    print(f"Logits B shape: {lb.shape}")
    
    # Check ensemble
    preds = model.get_ensemble_prediction(dummy_input)
    print(f"Ensemble Probs shape: {preds.shape}")
