# cache

This folder is the cache directory for Hugging Face (HF).

When using online mode, downloaded models will be cached in this folder.

For [offline mode](https://huggingface.co/docs/transformers/main/installation#offline-mode) use, please download the models in advance and specify the model directory,
such as the `facebook/watermark-anything` model below.

The folder structure for `./cache/huggingface/hub/models--facebook--watermark-anything` is as follows.

```
.
├── blobs
│   ├── a3270a567d669c53d2784143951749d4-10
│   ├── a6344aac8c09253b3b630fb776ae94478aa0275b
│   └── ba250a522a9eaada9332e3d5a3ce7e7d812b41c0
├── refs
│   └── main
└── snapshots
    └── f39387cd46bf657e553aacd52807e4442690eab8
        ├── checkpoint.pth -> ../../blobs/a3270a567d669c53d2784143951749d4-10
        ├── .gitattributes -> ../../blobs/a6344aac8c09253b3b630fb776ae94478aa0275b
        └── README.md -> ../../blobs/ba250a522a9eaada9332e3d5a3ce7e7d812b41c0

5 directories, 7 files
```

It will use
- `./cache/huggingface/hub/models--facebook--watermark-anything/snapshots/?`

For more details, refer to [up@cpu-offline/docker-compose.yml](./../docker/up@cpu-offline/docker-compose.yml).


## Pre-download for offline mode

Running in online mode will automatically download the model.

install cli

```bash
pip install -U "huggingface_hub[cli]"
```

download model

```bash
huggingface-cli download facebook/watermark-anything --repo-type model --revision main --cache-dir ./cache/huggingface/hub
```