FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Reachable from the browser on the HOST, not the Docker network — see
# src/api.js. Matches this Dockerfile's own image being run via
# ../docker-compose.yml's "3001:3001" port mapping for the api service.
ENV VITE_API_URL=http://localhost:3001

EXPOSE 3000
CMD ["npm", "run", "dev"]
