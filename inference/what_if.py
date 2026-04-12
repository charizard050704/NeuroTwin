import torch

def predict_future(model, input_seq, steps=10):
    model.eval()
    seq = input_seq.clone()
    preds = []

    for _ in range(steps):
        with torch.no_grad():
            pred, _, _ = model(seq.unsqueeze(0), target_len=1)

        pred = pred.squeeze()
        preds.append(pred.item())

        pred = pred.view(1,1)
        seq = torch.cat((seq[1:], pred), dim=0)

    return preds

def what_if_simulation(model, seq, factor=1.2, steps=10):
    noise = torch.randn_like(seq) * 0.02
    trend = torch.linspace(0, 0.05, steps=seq.shape[0]).unsqueeze(1)

    modified_seq = seq.clone() + noise + trend
    modified_seq = torch.clamp(modified_seq, 0, 1)

    orig = predict_future(model, seq, steps)
    mod = predict_future(model, modified_seq, steps)

    return orig, mod, modified_seq