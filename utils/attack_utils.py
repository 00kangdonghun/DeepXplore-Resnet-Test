import torch

def generate_adversarial(model, x, y, step_size=0.01, epsilon=0.1, steps=10):
    x_adv = x.clone().detach().requires_grad_(True)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(steps):
        output = model(x_adv)
        loss = loss_fn(output, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data.sign()
        x_adv = x_adv + step_size * grad
        perturbation = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + perturbation, 0, 1).detach().requires_grad_(True)

    return x_adv.detach()