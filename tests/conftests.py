from kgate.utils import set_seeds

def pytest_sessionstart(session):
    set_seeds()