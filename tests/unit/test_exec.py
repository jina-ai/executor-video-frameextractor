from executor import VideoFrameExtractor
from jina import Document, DocumentArray


def test_encode(tmp_path):
    workspace = str(tmp_path / 'workspace')
    encoder = VideoFrameExtractor(
        metas={
            'workspace': workspace
        }
    )
    docs = DocumentArray([
        Document(
        id = '2c2OmN49cj8.mp4',
        uri='tests/toy_data/2c2OmN49cj8.mp4'
    )])
    encoder.extract(docs=docs)
    assert len(docs[0].chunks) == 15
    for c in docs[0].chunks:
        assert len(c.blob.shape) == 3
