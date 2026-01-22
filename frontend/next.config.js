/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // Configure webpack for ONNX Runtime and WebAssembly support
  webpack: (config, { isServer }) => {
    // Enable WebAssembly experiments
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };
    
    // Handle .wasm files
    config.module.rules.push({
      test: /\.wasm$/,
      type: 'asset/resource',
    });
    
    // Ignore .node binary files
    config.module.rules.push({
      test: /\.node$/,
      loader: 'ignore-loader',
    });
    
    // Client-side specific config
    if (!isServer) {
      // Disable Node.js built-ins for browser
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
        module: false,
        perf_hooks: false,
      };
      
      // Externalize native modules - they won't be bundled
      config.externals = [
        ...config.externals || [],
        'onnxruntime-node',
        'sharp',
      ];
    }
    
    return config;
  },
  
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.API_URL ? `${process.env.API_URL}/:path*` : 'http://localhost:8000/api/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
