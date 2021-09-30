__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from pathlib import Path

import pytest
from jina import Document, DocumentArray, Executor

from executor import VideoFrameExtractor


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.fps == 1
    assert ex.max_num_frames == 50
    assert ex.debug == False


def test_no_docucments(encoder: VideoFrameExtractor):
    docs = DocumentArray()
    encoder.extract(docs=docs)
    assert len(docs) == 0  # SUCCESS


def test_none_docs(encoder: VideoFrameExtractor):
    encoder.extract(docs=None, parameters={})


def test_docs_no_uris(encoder: VideoFrameExtractor):
    docs = DocumentArray([Document()])

    with pytest.raises(ValueError, match='No uri'):
        encoder.extract(docs=docs, parameters={})

    assert len(docs) == 1
    assert len(docs[0].chunks) == 0


def test_encode(encoder: VideoFrameExtractor):
    docs = DocumentArray(
        [Document(id='2c2OmN49cj8.mp4', uri='tests/toy_data/2c2OmN49cj8.mp4')]
    )
    encoder.extract(docs=docs)
    assert len(docs[0].chunks) == 15
    for c in docs[0].chunks:
        assert len(c.blob.shape) == 3


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_encode(encoder: VideoFrameExtractor, batch_size: int):
    docs = DocumentArray(
        [
            Document(id=f'2c2OmN49cj8_{idx}.mp4', uri='tests/toy_data/2c2OmN49cj8.mp4')
            for idx in range(batch_size)
        ]
    )
    encoder.extract(docs=docs)

    for doc in docs:
        assert len(doc.chunks) == 15
        for c in doc.chunks:
            assert len(c.blob.shape) == 3
