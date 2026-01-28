FROM node:20-slim

WORKDIR /app

COPY package.json pnpm-lock.yaml* ./

RUN npm install -g pnpm

RUN pnpm install

COPY . .

# Expose Vite port
EXPOSE 5173

CMD ["pnpm", "run", "dev", "--", "--host"]
