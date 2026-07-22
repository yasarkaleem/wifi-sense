FROM node:20-slim

WORKDIR /app

# better-sqlite3 and zeromq are native modules; both ship prebuilt binaries
# for common platforms, but fall back to compiling from source if none
# matches the build target, which needs these — node:20-slim doesn't
# include them by default.
RUN apt-get update && apt-get install --no-install-recommends -y python3 make g++ \
    && rm -rf /var/lib/apt/lists/*

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY src ./src

ENV API_SUB_HOST=pipeline \
    API_SUB_PORT=5568 \
    API_HTTP_HOST=0.0.0.0 \
    API_HTTP_PORT=3001 \
    API_DB_PATH=/data/wifi-sense.db \
    API_CORS_ORIGIN=*

EXPOSE 3001

CMD ["npm", "start"]
