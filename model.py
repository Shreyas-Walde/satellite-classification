# """
# Model Architecture Definition
# This must match EXACTLY what was used during training.
# """
# import torch
# import torch.nn as nn


# class ImageClassificationBase(nn.Module):
#     """Base class for image classification models"""
    
#     def training_step(self, batch):
#         images, labels = batch
#         out = self(images)
#         loss = nn.functional.cross_entropy(out, labels)
#         return loss
    
#     def validation_step(self, batch):
#         images, labels = batch
#         out = self(images)
#         loss = nn.functional.cross_entropy(out, labels)
#         acc = accuracy(out, labels)
#         return {'val_loss': loss.detach(), 'val_acc': acc}
    
#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
#     def epoch_end(self, epoch, result):
#         print(f"Epoch [{epoch}], last_lr: {result['lrs'][-1]:.5f}, "
#               f"train_loss: {result['train_loss']:.4f}, "
#               f"val_loss: {result['val_loss']:.4f}, "
#               f"val_acc: {result['val_acc']:.4f}")


# def conv_block(in_channels, out_channels, pool=False):
#     """Convolutional block with optional pooling"""
#     layers = [
#         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     ]
#     if pool:
#         layers.append(nn.MaxPool2d(2))
#     return nn.Sequential(*layers)


# class ResNet9(ImageClassificationBase):
#     """
#     ResNet9 Architecture - MUST match training exactly!
    
#     Input: 3 channels (RGB)
#     Output: 4 classes (cloudy, desert, green_area, water)
#     """
    
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
        
#         # First block
#         self.conv1 = conv_block(in_channels, 64)
        
#         # Second block with pooling
#         self.conv2 = conv_block(64, 128, pool=True)
#         self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
#         # Third block with pooling
#         self.conv3 = conv_block(128, 256, pool=True)
        
#         # Fourth block with pooling
#         self.conv4 = conv_block(256, 512, pool=True)
#         self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.AdaptiveMaxPool2d(1),
#             nn.Flatten(),
#             nn.Dropout(0.2),
#             nn.Linear(512, num_classes)
#         )
    
#     def forward(self, xb):
#         """Forward pass through the network"""
#         out = self.conv1(xb)
#         out = self.conv2(out)
#         out = self.res1(out) + out  # Residual connection
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.res2(out) + out  # Residual connection
#         out = self.classifier(out)
#         return out


# def accuracy(outputs, labels):
#     """Calculate accuracy"""
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# def load_model(model_path: str, device: str = 'cpu'):
#     """
#     Load the trained model from checkpoint
    
#     Args:
#         model_path: Path to the .pth file
#         device: 'cpu' or 'cuda'
    
#     Returns:
#         Loaded model in eval mode
#     """
#     # Initialize model with same architecture as training
#     model = ResNet9(in_channels=3, num_classes=4)
    
#     try:
#         # Load the checkpoint
#         checkpoint = torch.load(model_path, map_location=device)
        
#         # Check if it's a full checkpoint or just state_dict
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#             print("✅ Loaded model from checkpoint with metadata")
#             if 'classes' in checkpoint:
#                 print(f"   Classes: {checkpoint['classes']}")
#         else:
#             model.load_state_dict(checkpoint)
#             print("✅ Loaded model state dict directly")
        
#         # Set to evaluation mode
#         model.eval()
#         model.to(device)
        
#         print(f"✅ Model loaded successfully on {device}")
#         return model
        
#     except Exception as e:
#         print(f"❌ Error loading model: {e}")
#         raise


# if __name__ == "__main__":
#     # Test model loading
#     print("Testing model architecture...")
#     model = ResNet9(3, 4)
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
#     # Test forward pass
#     dummy_input = torch.randn(1, 3, 64, 64)
#     output = model(dummy_input)
#     print(f"Output shape: {output.shape}")
#     print("✅ Model architecture test passed!")