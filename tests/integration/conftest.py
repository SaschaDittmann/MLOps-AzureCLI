import pytest

def pytest_addoption(parser):
    parser.addoption("--scoreurl", action="store",
        help="the score url of the ml web service")

@pytest.fixture
def scoreurl(request):
    return request.config.getoption("--scoreurl")
