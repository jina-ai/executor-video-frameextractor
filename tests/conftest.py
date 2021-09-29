import pytest
from jina import Document, DocumentArray

from executor import VideoFrameExtractor


@pytest.fixture()
def encoder(tmp_path) -> VideoFrameExtractor:
    workspace = str(tmp_path / 'workspace')
    encoder = VideoFrameExtractor(metas={'workspace': workspace})
    return encoder


@pytest.fixture(scope='package')
def build_da():
    def _build_da():
        return DocumentArray(
            [Document(id='2c2OmN49cj8.mp4', uri='tests/toy_data/2c2OmN49cj8.mp4')]
        )

    return _build_da
