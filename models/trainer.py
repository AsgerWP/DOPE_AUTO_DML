import torch
from datasets.datasets import ATEDataset
import copy


def train_model(model, data: ATEDataset, loss_fn, lr, weight_decay, batch_size, epochs, patience, device):
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    train_data, test_data = data.split_into_train_and_test_sets(train_size=0.8)
    train_loader = train_data.create_dataloader(batch_size=batch_size)
    test_loader = test_data.create_dataloader(batch_size=batch_size)
    best = 1e6
    counter = 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            batch = tuple(x.to(device) for x in batch)
            loss = loss_fn(batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for batch in test_loader:
                batch = tuple(x.to(device) for x in batch)
                test_loss += loss_fn(batch).item()
            if test_loss < best:
                best = test_loss
                counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter == patience:
                    model.load_state_dict(best_state)
                    break
