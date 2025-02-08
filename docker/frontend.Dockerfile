FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build app
RUN npm run build

EXPOSE 8000

# Start the app using npm
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "8000"]