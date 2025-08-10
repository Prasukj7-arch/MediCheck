# Medical Report RAG Extractor - Next.js Version

A modern, animated React/Next.js application for extracting insights from medical reports using Retrieval-Augmented Generation (RAG) technology.

## 🚀 Features

### ✨ Modern React Architecture
- **TypeScript** for type safety and better development experience
- **Next.js 14** with App Router for optimal performance
- **Framer Motion** for smooth animations and transitions
- **React Dropzone** for intuitive file uploads
- **Lucide React** for beautiful, consistent icons

### 🎨 Enhanced User Experience
- **Dual Perspectives**: Patient and Doctor views with specialized interfaces
- **Smooth Animations**: Page transitions, button interactions, and loading states
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Real-time Status**: System health monitoring with automatic updates
- **Error Handling**: Comprehensive error states with user-friendly messages

### 🔧 Technical Enhancements
- **API Service Layer**: Centralized HTTP client with axios interceptors
- **Type Safety**: Full TypeScript coverage with custom interfaces
- **State Management**: React hooks for efficient state handling
- **CSS Modules**: Scoped styling for maintainable code
- **Error Boundaries**: Graceful error handling throughout the app

## 📁 Project Structure

```
nextjs-version/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   └── page.tsx            # Main page component
│   ├── components/
│   │   └── MedicalReportExtractor.tsx  # Main component
│   ├── services/
│   │   └── api.ts              # API service layer
│   ├── styles/
│   │   └── MedicalReportExtractor.module.css  # Styling
│   └── types/
│       └── index.ts            # TypeScript definitions
├── package.json
├── tsconfig.json
├── next.config.js
└── README.md
```

## 🛠️ Installation & Setup

### Prerequisites
- Node.js 18+ 
- npm or yarn
- The Flask backend running on `http://localhost:5000`

### Installation

1. **Navigate to the Next.js directory:**
   ```bash
   cd nextjs-version
   ```

2. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Set up environment variables:**
   Create a `.env.local` file:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:5000
   ```

4. **Start the development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

5. **Open your browser:**
   Navigate to `http://localhost:3000`

## 🎯 Key Features

### Patient Perspective
- **Simple Language**: Easy-to-understand explanations
- **Reassuring Tone**: Supportive responses for health concerns
- **Practical Advice**: Lifestyle recommendations and monitoring tips
- **Example Questions**: Pre-built queries for common patient concerns

### Doctor Perspective
- **Clinical Analysis**: Detailed medical terminology and findings
- **Differential Diagnosis**: Professional diagnostic considerations
- **Treatment Protocols**: Evidence-based recommendations
- **Red Flag Detection**: Identification of concerning findings

### Technical Features
- **Drag & Drop Upload**: Intuitive PDF file handling
- **Real-time Status**: Live system health monitoring
- **Error Recovery**: Graceful handling of network issues
- **Loading States**: Visual feedback during operations
- **Responsive Design**: Optimized for all screen sizes

## 🔧 Configuration

### Environment Variables
- `NEXT_PUBLIC_API_URL`: Backend API URL (default: http://localhost:5000)

### API Endpoints
The component expects the following Flask backend endpoints:
- `POST /upload` - File upload endpoint
- `POST /query` - Query processing endpoint
- `GET /status` - System status endpoint

## 🎨 Customization

### Styling
- Modify `src/styles/MedicalReportExtractor.module.css` for visual changes
- CSS modules provide scoped styling to prevent conflicts
- Responsive breakpoints included for mobile optimization

### Animations
- Framer Motion animations can be customized in the component
- Animation variants are defined for consistent motion
- Performance optimized with `AnimatePresence`

### Adding New Features
1. **New API endpoints**: Add to `src/services/api.ts`
2. **New types**: Extend `src/types/index.ts`
3. **New components**: Create in `src/components/`
4. **New styles**: Add to CSS modules

## 🚀 Deployment

### Vercel (Recommended)
```bash
npm run build
# Deploy to Vercel with automatic builds
```

### Other Platforms
```bash
npm run build
npm start
```

## 🔍 Development

### Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

### Code Quality
- TypeScript for type safety
- ESLint for code linting
- Prettier for code formatting (recommended)

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Ensure Flask backend is running on port 5000
   - Check `NEXT_PUBLIC_API_URL` environment variable
   - Verify CORS settings on backend

2. **Build Errors**
   - Clear `.next` folder: `rm -rf .next`
   - Reinstall dependencies: `npm install`
   - Check TypeScript errors: `npx tsc --noEmit`

3. **Animation Issues**
   - Ensure Framer Motion is properly installed
   - Check for conflicting CSS animations
   - Verify component mounting/unmounting

## 📈 Performance

### Optimizations
- **Code Splitting**: Automatic with Next.js
- **Image Optimization**: Built-in Next.js image optimization
- **Bundle Analysis**: Use `@next/bundle-analyzer`
- **Lazy Loading**: Components load on demand

### Monitoring
- **Error Tracking**: Integrate with Sentry or similar
- **Performance Monitoring**: Use Vercel Analytics or similar
- **User Analytics**: Google Analytics or similar

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆚 Comparison with Original

### Improvements
- **Modern Stack**: React 18 + Next.js 14 vs plain HTML
- **Type Safety**: TypeScript vs JavaScript
- **Animations**: Framer Motion vs CSS only
- **Error Handling**: Comprehensive vs basic
- **State Management**: React hooks vs vanilla JS
- **File Upload**: React Dropzone vs manual handling
- **Icons**: Lucide React vs emoji/text
- **Responsive**: Mobile-first design
- **Performance**: Optimized bundle and loading

### Maintained Features
- **Dual Perspectives**: Patient and Doctor views
- **RAG Integration**: Same backend API
- **File Upload**: PDF processing
- **Query System**: Question-answer functionality
- **System Status**: Health monitoring
- **Example Questions**: Pre-built queries

The Next.js version provides a significantly enhanced user experience while maintaining all the core functionality of the original application.
