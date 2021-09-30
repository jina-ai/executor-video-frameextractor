import pytest

from executor import VideoFrameExtractor


@pytest.fixture()
def encoder(tmp_path) -> VideoFrameExtractor:
    workspace = str(tmp_path / 'workspace')
    encoder = VideoFrameExtractor(metas={'workspace': workspace})
    return encoder
