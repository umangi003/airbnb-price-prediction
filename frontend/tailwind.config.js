/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'cherry-red': '#C41E3A',
        'eggshell': '#F0EFE6',
        'slate-gray': '#6B7280',
      },
      fontFamily: {
        sans: ['Poppins', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
      },
      borderRadius: {
        'xs': '8px',
        'sm': '12px',
      },
      spacing: {
        'generous': '24px',
      },
      maxWidth: {
        'form': '42rem',
      },
    },
  },
  plugins: [],
}