__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from jina import Document, DocumentArray, Flow

from executor import VideoFrameExtractor


def test_integration(tmp_path):
    da = DocumentArray(
        [Document(id='2c2OmN49cj8.mp4', uri='tests/toy_data/2c2OmN49cj8.mp4')]
    )
    workspace = str(tmp_path / 'workspace')
    with Flow().add(
        uses=VideoFrameExtractor, uses_metas={'workspace': workspace}
    ) as flow:
        resp = flow.post(on='/index', inputs=da, return_results=True)

    assert len(resp[0].docs) == 1
    for doc in resp[0].docs:
        assert len(doc.chunks) == 15
        for c in doc.chunks:
            assert len(c.blob.shape) == 3
