#!/bin/sh
set -e
if [ -z "$UI_USER" ] || [ -z "$UI_PASSWORD" ]; then
  echo "UI_USER and UI_PASSWORD must be set" >&2; exit 1
fi
printf '%s:%s\n' "$UI_USER" "$(openssl passwd -apr1 "$UI_PASSWORD")" > /etc/nginx/.htpasswd
exec nginx -g 'daemon off;'
