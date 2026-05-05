# ──────────────────────────────────────────────────────────────────
# StackMind — Frontend Dockerfile (Nginx)
# Builds the Vite app and serves it behind Nginx with API proxy.
# ──────────────────────────────────────────────────────────────────

# ── Stage 1: Build ────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /build

COPY package.json ./
RUN npm install

COPY . .
RUN npm run build

# ── Stage 2: Serve ────────────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/stackmind.conf

# Copy built frontend
COPY --from=builder /build/dist /usr/share/nginx/html

# Expose HTTP
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
