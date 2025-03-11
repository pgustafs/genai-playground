# Nginx Deployment for Gradio with Podman on RHEL

## Overview
The configuration provides:

- HTTP and HTTPS servers
- Environment variable support through a templating system
- WebSocket support for Gradio's real-time functionality
- Proper proxy headers for forwarding requests
- Compatible with RHEL 9 and Podman containerization

## Environment Variables
|**Variable**|**Description**|**Default Value**|
|------------|---------------|-----------------|
|`SERVER_NAME` |The server name (domain)|`localhost`|
|`GRADIO_LOCATION` |The URL path where Gradio is served|`/`|
|`GRADIO_UPSTREAM` |The Gradio application URL|`http://gradio:7860`|
|`SSL_CERT_PATH` |Path to SSL certificate|`/etc/nginx/ssl/nginx.crt`|
|`SSL_KEY_PATH` |Path to SSL private key|`/etc/nginx/ssl/nginx.key`|

## How It Works

1. The system uses a template Nginx configuration file with placeholders
2. When the container starts, the entrypoint script:
    - Reads environment variables
    - Replaces placeholders in the template with actual values
    - Generates a finalized nginx.conf
    - Starts Nginx with this configuration

This approach allows for dynamic configuration without requiring Nginx to directly support environment variables.

## Deployment Steps

### Cloning the genai-playground Repository

To clone the genai-playground repository, which contains various concepts and test applications for generative AI, use the following command:

```bash
git clone https://github.com/pgustafs/genai-playground.git
```

### Building the Nginx-Gradio Proxy Container Image

To create a Container image for the Nginx-Gradio-Proxy, you can use the `podman build` command. This command will build the Container image using the Containerfile present in the current directory (denoted by the `.`).

```bash
cd genai-playground/nginx-gradio-proxy
```

```bash
podman build -t nginx-gradio-proxy .
```