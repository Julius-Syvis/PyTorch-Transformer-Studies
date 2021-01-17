import torch


# Saving / loading state

def save(model, optimizer, path):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, path)


def load(path):
    state = torch.load(path)
    return state['model'], state['optimizer']


# Saving / loading whole model

def save_whole(model, optimizer, path):
    state = {"model": model, "optimizer": optimizer}
    torch.save(state, path)


def load_whole(path):
    state = torch.load(path)
    return state['model'], state['optimizer']
