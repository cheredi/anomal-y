import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MultiLayerGradCAMResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.target_layers = ['layer2', 'layer3', 'layer4']
        self.gradients = {}
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        for name in self.target_layers:
            layer = dict(self.model.named_modules())[name]
            layer.register_forward_hook(self._save_activations(name))
            layer.register_full_backward_hook(self._save_gradients(name))

    def _save_activations(self, name):
        def hook(module, input, output):
            self.activations[name] = output
        return hook

    def _save_gradients(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0]
        return hook

    def forward(self, x):
        return self.model(x)

    def generate_fused_cam(self, input_tensor, class_idx=None):
        self.zero_grad()
        output = self.forward(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1

        output.backward(gradient=one_hot, retain_graph=True)

        cams = []
        for name in self.target_layers:
            grad = self.gradients[name]
            act = self.activations[name]

            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * act, dim=1)
            cam = F.relu(cam)

            # Normalize CAM
            cam_min = cam.view(cam.size(0), -1).min(1, keepdim=True)[0].view(-1, 1, 1)
            cam_max = cam.view(cam.size(0), -1).max(1, keepdim=True)[0].view(-1, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            cam = F.interpolate(cam.unsqueeze(1), size=(224, 224), mode='bilinear',
                                align_corners=False).squeeze(1)
            cams.append(cam)

        fused_cam = torch.stack(cams).mean(0)
        return fused_cam.squeeze(0) 
