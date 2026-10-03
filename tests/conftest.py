def pytest_sessionstart(session):
    # Tiny mechanics benefit from one CPU thread; no real-model runs in this suite.
    # torch is optional: the stdlib-only evidence-plane suite runs without it.
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
