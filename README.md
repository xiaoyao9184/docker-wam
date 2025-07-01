# Docker watermark-anything

A Docker image built through Github Actions with Git commit version tag

[![Docker Image Build/Publish tag with commit](https://github.com/xiaoyao9184/docker-wam/actions/workflows/docker-image-tag-commit.yml/badge.svg)](https://github.com/xiaoyao9184/docker-wam/actions/workflows/docker-image-tag-commit.yml) [![](https://img.shields.io/docker/v/xiaoyao9184/watermark-anything)](https://hub.docker.com/r/xiaoyao9184/watermark-anything)

[![HuggingFace Space Sync](https://github.com/xiaoyao9184/docker-wam/actions/workflows/hf-space-sync.yml/badge.svg)](https://github.com/xiaoyao9184/docker-wam/actions/workflows/hf-space-sync.yml) [![](https://img.shields.io/badge/HuggingFace-space-ff9f44?logo=huggingface)](https://huggingface.co/spaces/xiaoyao9184/watermark-anything) ![](https://badge.mcpx.dev?type=server 'MCP Server')

![LabelStudio ML](https://img.shields.io/badge/LabelStudio-ML-ff7557?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA%2FwD%2FAP%2BgvaeTAAABhElEQVRIie2UzStEURjGf%2Bd%2BIFkoNc0sWAgLZUOS9ZSVBQvGwspCliym%2FAdTsiQpG1lpVuIfYMPCSKMoFkr5HBFFGM5rccPljnOnu%2BWps3ju%2Bzzv73bqXiXpoTl%2BSmRXTS%2FPB577I%2BlUD6j%2B4OBtSk1njz%2BsAzIWCClWACMAaC%2FZtaxFwA8oqS5Jp1ZDAE0hcyMgDvSWsyAaIFEPnUlz82gPDnIRAW4V1MXMzdOa0OUewNExNHG0lacuBj2DYP12cz61dkBLG%2BQ24DAPihFsvUZl4fYbQGWyBZkYqMABlA1V1WW9GY7jHdv1vMi9ymQLP2NWedui6x%2FwFwAyPpxA2d5XpTU8PsDLU3jztehl34qeF6tWxocTMjrq%2BmMObvHs091cwvIsNDRDMvgn%2Fqb9Hcitf3klC7hFqL3rBra%2BAKX08gzX52bA4715bgRcnMDqUlkLogGUusJ2d4xN%2FdqI1i3RACKbKrPUZypKOjUJZMIBwkyJej6siMg2qGDXVheh3X%2F59Q535W%2Fus0NULAAAAABJRU5ErkJggg%3D%3D)

# Why

I found that watermark-anything's Docker image is difficult to find.
The code on [GitHub](https://github.com/facebookresearch/watermark-anything) does not provide a pre-built Docker image.

This project will use GitHub Actions and Docker Hub to build and publish images,
aiming to keep the process as clean as possible without custom configuration files.

# Tags

The images of this project will be published to Docker Hub under the repository [xiaoyao9184/watermark-anything](https://hub.docker.com/r/xiaoyao9184/watermark-anything).

Since this project references the watermark-anything project via a submodule, it cannot monitor push events on the watermark-anything project, and therefore cannot automatically create an image for every commit.
A good solution is to manually trigger the action and tag it with the commit id. For more details, see this article [set-dynamic-parameters-github-workflows-en](https://damienaicheh.github.io/github/actions/2022/01/20/set-dynamic-parameters-github-workflows-en.html).

The default image name format is `${DOCKERHUB_USERNAME}/watermark-anything`.

The tag uses the input parameter `commit_id`,
which can be either a branch name or a commit id, 
when manually triggering the [docker-image-tag-commit](./.github/workflows/docker-image-tag-commit.yml) job.
if the job is triggered by a submodule update push,
the default branch name `master` will be used instead of the `commit_id` parameter.
This job will also use the shortened commit id as the tag.

Currently, only the `linux/amd64` platform is supported.

# Model

The models of this project will be synced to HuggingFace under the model [xiaoyao9184/watermark-anything](https://huggingface.co/xiaoyao9184/watermark-anything).

The Docker image does not include model files.
When running, the required models will be automatically downloaded.

If you need to run offline, you must pre-download the model files and enable offline mode.
See [cache/README.md](./cache/README.md) for detailed instructions.

# Service

By default, the Docker container runs the Jupyter Notebook, which comes from the original project.

However, this project also provides a Streamlit App and a Gradio App.
The Gradio App supports both a UI and API interface, and can even serve as an MCP server,
so it is recommended as the preferred option.

The source code for the Gradio App is located in the [gradio](./gradio) directory of this project.
A demo of this project is also available and auto-synced on Hugging Face Spaces: [xiaoyao9184/watermark-anything](https://huggingface.co/spaces/xiaoyao9184/watermark-anything)

To run the Gradio App, you can do so by modifying the Docker command. see the `up.gradio` sub-directory in the [docker](./docker) directory for details.

And this project also provides Label Studio ML Backend.

The source code for the Label Studio ML Backend is located in the [label](./label) directory of this project.

To run the Label Studio ML Backend, you can do so by modifying the Docker command. see the `up.label` sub-directory in the [docker](./docker) directory for details.

# Change

You can fork this project and build your own image. You will need to provide the following variables: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `HF_USERNAME`, `HF_TOKEN`.
See [this](https://github.com/docker/login-action#docker-hub) for more details.
