<!--
---
title: Watermark massage from images with WatermarkAnything
type: guide
tier: all
order: 101
hide_menu: true
hide_frontmatter_title: true
meta_title: WatermarkAnything model connection for detect watermark in images
meta_description: The WatermarkAnything model connection integrates the capabilities of WatermarkAnything with Label Studio to assist in machine learning labeling tasks involving Watermark detect.
categories:
    - Computer Vision
    - WatermarkAnything
image: "/tutorials/wam.png"
---
-->

# WatermarkAnything model connection

The [WatermarkAnything](https://github.com/facebookresearch/watermark-anything) model connection is a powerful tool that integrates the capabilities of WatermarkAnything with Label Studio. It is designed to assist in machine learning labeling tasks, specifically those involving watermark detection.

The primary function of this connection is to recognize and extract watermark information from images, which is a crucial step in many machine learning workflows. By automating this process, the WatermarkAnything model connection can significantly increase efficiency, reducing the time and effort required for manually extracting watermark text.

In the context of Label Studio, this connection enhances the platform's labeling capabilities, allowing users to automatically generate watermark labels for images. This is particularly useful for tasks such as copyright checks and AI-generated fake image labeling.

## Before you begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart).

This tutorial uses the [`wam` example](https://github.com/xiaoyao9184/docker-wam/tree/main/label).

## Labeling configuration

The WatermarkAnything model connection needs to be used with the labeling configuration in Label Studio. The configuration uses the following labels:

```xml
<View>
  <Image name="image" value="$image"/>

  <Header value="watermark label:"/>
  <BrushLabels name="watermark_mask" toName="image">
    <Label value="watermarked" />
  </BrushLabels>

  <TextArea name="watermark_msg" toName="image"
            maxSubmissions="1"
            editable="false"
            displayMode="region-list"
            rows="1"
            required="true"
            perRegion="true"
            />
</View>
```


> Warning! Please note that the current implementation of the WatermarkAnything model connection does not support images that are directly uploaded to Label Studio. It is designed to work with images that are hosted publicly on the internet. Therefore, to use this connection, you should ensure that your images are publicly accessible via a URL.


## Running with Docker (recommended)

1. Start the Machine Learning backend on `http://localhost:9090` with the prebuilt image:

```bash
cd docker/up.label@gpu-online
docker-compose up
```

2. Validate that backend is running

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Create a project in Label Studio. Then from the **Model** page in the project settings, [connect the model](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio). The default URL is `http://localhost:9090`.


## Building from source (advanced)

To build the ML backend from source, you have to clone the repository and build the Docker image:

```bash
docker build -t xiaoyao9184/wam:main -f ./docker/build@source/dockerfile .
```

## Running without Docker (advanced)

To run the ML backend without Docker, you have to clone the repository and install all dependencies using conda:

```bash
conda env create -f ./environment.yml
```

Then you can start the ML backend:

```bash
conda activate wam
label-studio-ml start --root-dir . label
```

The WatermarkAnything model connection offers several configuration options that can be set in the `docker-compose.yml` file:

- `BASIC_AUTH_USER`: Specifies the basic auth user for the model server.
- `BASIC_AUTH_PASS`: Specifies the basic auth password for the model server.
- `LOG_LEVEL`: Sets the log level for the model server.
- `WORKERS`: Specifies the number of workers for the model server.
- `THREADS`: Specifies the number of threads for the model server.
- `MODEL_DIR`: Specifies the model directory.
- `LABEL_STUDIO_ACCESS_TOKEN`: Specifies the Label Studio access token.
- `LABEL_STUDIO_HOST`: Specifies the Label Studio host.

These options allow you to customize the behavior of the WatermarkAnything model connection to suit your specific needs.

# Customization

The ML backend can be customized by adding your own models and logic inside the `./label` directory. 