# VideoFrameExtractor
**FrameExtractor** is an executor that extracts frames from videos with `ffmpeg`


## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://VideoFrameExtractor')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://VideoFrameExtractor')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
