import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from models import resnet50_a, resnet50_b
from data.cifar10_loader import get_cifar10
from utils.neuron_coverage import NeuronCoverage
from utils.attack_utils import generate_adversarial

# Prepare environment
os.makedirs("logs", exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
model1 = resnet50_a.get_model().to(device).eval()
model2 = resnet50_b.get_model().to(device).eval()
cov1 = NeuronCoverage(model1)
cov2 = NeuronCoverage(model2)

# Load data
loader = get_cifar10(batch_size=1)

success_count = 0
max_success = 30

for idx, (x, y) in enumerate(loader):
    if success_count >= max_success:
        break

    x, y = x.to(device), y.to(device)
    cov1.reset()
    cov2.reset()

    # Inference on original
    out1 = model1(x)
    out2 = model2(x)
    pred1 = torch.argmax(out1, dim=1)
    pred2 = torch.argmax(out2, dim=1)

    if pred1 != pred2:
        print(f"[Divergence] idx={idx} model1={pred1.item()} model2={pred2.item()} true={y.item()}")
        save_image(x.cpu(), f"logs/diverge_{idx}_orig.png")

        _ = model1(x)
        _ = model2(x)
        print(f"Neuron Coverage - model1: {cov1.compute_coverage():.3f}, model2: {cov2.compute_coverage():.3f}")

        # Generate adversarial
        x_adv = generate_adversarial(model1, x, y)
        pred1_adv = torch.argmax(model1(x_adv), dim=1)
        pred2_adv = torch.argmax(model2(x_adv), dim=1)

        if pred1_adv != pred2_adv:
            save_image(x_adv.cpu(), f"logs/diverge_{idx}_adv.png")
            print(f"[Attack Success] idx={idx} model1_adv={pred1_adv.item()} model2_adv={pred2_adv.item()}")
            success_count += 1

print(f"\nTotal successful adversarial divergences: {success_count}")
