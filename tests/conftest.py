import torch


def pytest_sessionstart(session):
    # Tiny mechanics benefit from one CPU thread; no real-model runs in this suite.
    torch.set_num_threads(1)
