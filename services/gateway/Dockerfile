# Build context: repository root. Config-only gateway: static UI + reverse proxy, runs as non-root.
FROM nginxinc/nginx-unprivileged:1.29.3-alpine
ENV GATEWAY_PORT=8000 MAX_UPLOAD_MB=200 PUBLIC_ORIGIN=http://localhost:8000 TRUSTED_PROXY_CIDR=127.0.0.1/32
COPY --chmod=755 services/gateway/public-host.envsh /docker-entrypoint.d/16-public-host.envsh
COPY services/gateway/default.conf.template /etc/nginx/templates/default.conf.template
COPY services/gateway/static /usr/share/nginx/html
EXPOSE 8000
