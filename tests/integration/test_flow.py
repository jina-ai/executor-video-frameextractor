__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from jina import Flow

from executor import VideoFrameExtractor


def test_integration(tmp_path, build_da):
    da = build_da()
    workspace = str(tmp_path / 'workspace')
    with Flow().add(uses=VideoFrameExtractor, metas={'workspace': workspace}) as flow:
        resp = flow.post(on='/index', inputs=da, return_results=True)

    assert len(resp[0].docs) == 1
    for doc in resp[0].docs:
        assert len(doc.chunks) == 15
        for c in doc.chunks:
            assert len(c.blob.shape) == 3
