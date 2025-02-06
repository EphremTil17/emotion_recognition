# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY ../frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY ../frontend/ .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx config
COPY ../nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]