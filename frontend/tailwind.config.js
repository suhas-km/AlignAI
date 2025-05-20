/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    container: {
      center: true,
      padding: '2rem',
      screens: {
        '2xl': '1400px',
      },
    },
    extend: {
      colors: {
        border: 'hsl(0, 0%, 20%)',
        input: 'hsl(0, 0%, 15%)',
        ring: 'hsl(0, 0%, 30%)',
        background: 'hsl(0, 0%, 5%)',
        foreground: 'hsl(0, 0%, 95%)',
        primary: {
          DEFAULT: 'hsl(0, 0%, 100%)',
          foreground: 'hsl(0, 0%, 5%)',
        },
        secondary: {
          DEFAULT: 'hsl(0, 0%, 15%)',
          foreground: 'hsl(0, 0%, 95%)',
        },
        destructive: {
          DEFAULT: 'hsl(0, 85%, 60%)',
          foreground: 'hsl(0, 0%, 100%)',
        },
        muted: {
          DEFAULT: 'hsl(0, 0%, 10%)',
          foreground: 'hsl(0, 0%, 70%)',
        },
        accent: {
          DEFAULT: 'hsl(0, 0%, 20%)',
          foreground: 'hsl(0, 0%, 95%)',
        },
        popover: {
          DEFAULT: 'hsl(0, 0%, 10%)',
          foreground: 'hsl(0, 0%, 95%)',
        },
        card: {
          DEFAULT: 'hsl(0, 0%, 8%)',
          foreground: 'hsl(0, 0%, 95%)',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
    },
  },
  plugins: [require('tailwindcss-animate'), require('@tailwindcss/typography')],
  darkMode: 'class',
}
