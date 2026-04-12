import torch
import torch.nn as nn

def train_model(model, train_loader, X_val, y_val, epochs, lr):

    # ✅ Custom loss (CORRECT INDENTATION)
    def custom_loss(pred, target):
        mse = nn.MSELoss()(pred, target)

        grad_pred = pred[:, 1:] - pred[:, :-1]
        grad_true = target[:, 1:] - target[:, :-1]

        grad_loss = nn.MSELoss()(grad_pred, grad_true)

        return mse + 1.5 * grad_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 7
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:

            # ✅ Noise add
            X_batch = X_batch + torch.randn_like(X_batch) * 0.01

            # ✅ Multi-step prediction
            outputs, mu, logvar = model(X_batch, target_len=3)

            # ✅ Match target shape
            y_batch = y_batch.unsqueeze(1).repeat(1, 3, 1)

            # ✅ Loss
            recon_loss = custom_loss(outputs, y_batch)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + 0.07 * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # ✅ Validation
        model.eval()
        with torch.no_grad():
            val_preds, _, _ = model(X_val, target_len=1)
            val_loss = criterion(val_preds.squeeze(), y_val.squeeze())

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_loss/len(train_loader):.6f} | Val Loss: {val_loss.item():.6f}")

        # ✅ Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break