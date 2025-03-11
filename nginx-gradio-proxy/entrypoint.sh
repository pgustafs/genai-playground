#!/bin/bash
set -e

# Set default values for environment variables
SERVER_NAME=${SERVER_NAME:-localhost}
GRADIO_LOCATION=${GRADIO_LOCATION:-/}
GRADIO_UPSTREAM=${GRADIO_UPSTREAM:-http://127.0.0.1:7860}
SSL_CERT_PATH=${SSL_CERT_PATH:-/etc/nginx/ssl/nginx.crt}
SSL_KEY_PATH=${SSL_KEY_PATH:-/etc/nginx/ssl/nginx.key}

# Ensure GRADIO_LOCATION starts with / and ends with / if not just /
if [ "$GRADIO_LOCATION" != "/" ]; then
  GRADIO_LOCATION=${GRADIO_LOCATION#/}  # Remove leading / if present
  GRADIO_LOCATION="/${GRADIO_LOCATION}"  # Add leading /
  
  # Ensure trailing slash
  if [[ "$GRADIO_LOCATION" != */ ]]; then
    GRADIO_LOCATION="${GRADIO_LOCATION}/"
  fi
fi

echo "Configuring Nginx with the following settings:"
echo "Server Name: $SERVER_NAME"
echo "Gradio Location: $GRADIO_LOCATION"
echo "Gradio Upstream: $GRADIO_UPSTREAM"
echo "SSL Certificate: $SSL_CERT_PATH"
echo "SSL Key: $SSL_KEY_PATH"

# Create nginx config from template
cp /etc/nginx/nginx.conf.template /etc/nginx/nginx.conf

# Replace placeholders with actual values
sed -i "s|SERVER_NAME_PLACEHOLDER|$SERVER_NAME|g" /etc/nginx/nginx.conf
sed -i "s|GRADIO_LOCATION_PLACEHOLDER|$GRADIO_LOCATION|g" /etc/nginx/nginx.conf
sed -i "s|GRADIO_UPSTREAM_PLACEHOLDER|$GRADIO_UPSTREAM|g" /etc/nginx/nginx.conf
sed -i "s|SSL_CERT_PATH_PLACEHOLDER|$SSL_CERT_PATH|g" /etc/nginx/nginx.conf
sed -i "s|SSL_KEY_PATH_PLACEHOLDER|$SSL_KEY_PATH|g" /etc/nginx/nginx.conf

# Print the final configuration for debugging
echo "Generated Nginx configuration:"
echo "----------------------------------------"
cat /etc/nginx/nginx.conf
echo "----------------------------------------"

# Start Nginx
echo "Starting Nginx..."
exec nginx -g 'daemon off;'