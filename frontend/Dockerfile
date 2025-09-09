# Use Node 20 on Debian (no musl headaches)
FROM node:20-bullseye

WORKDIR /app

# Copy package files first (for cache efficiency)
COPY package*.json ./

# Install dependencies inside the container (no host node_modules!)
RUN rm -rf node_modules package-lock.json && npm install

# Copy the rest of the app
COPY . .

# Expose the dev server port
EXPOSE 5173

# Run Vite on 0.0.0.0 so host can reach it
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]


