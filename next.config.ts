import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_BACKEND_URL: process.env.NEXT_PUBLIC_BACKEND_URL,
  },
  turbopack: {
    resolveAlias: {
      canvas: 'false',
    },
  },
};

export default nextConfig;
