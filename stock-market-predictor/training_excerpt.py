"""Training/inference cell from stock_market_predictor.ipynb.

Run after its data preparation, model, and optimizer setup cells.
"""

# Dataset conversion is safe to repeat: NumPy arrays are retained as the source.
train_dataset = TimeSeriesDataset(x_train, y_train)
test_dataset = TimeSeriesDataset(x_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

history = []
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in train_dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x_batch)
        assert outputs.shape == y_batch.shape, (outputs.shape, y_batch.shape)
        loss = criterion(outputs, y_batch)  # Both have shape [batch, 1].
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(x_batch)
    epoch_loss = total_loss / len(train_dataset)
    history.append(epoch_loss)
    if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
        print(f'Epoch [{epoch + 1}/{epochs}], Mean training MAE (scaled): {epoch_loss:.6f}')

model.eval()
with torch.no_grad():
    y_pred_scaled = torch.cat([
        model(x_batch.to(device)).cpu() for x_batch, _ in test_dataloader
    ]).numpy()
y_pred = scaler_pred.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_pred.inverse_transform(test_dataset.y.numpy())
baseline_pred = close_values[train_data_len - 1:-1].copy()
assert y_pred.shape == y_test_unscaled.shape == baseline_pred.shape
assert np.isfinite(y_pred).all()
