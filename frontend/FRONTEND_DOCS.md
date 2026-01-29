# Frontend Application - Documentation

## 🏗️ Architecture

Built with **Next.js 14** (App Router), **React 18**, and **TypeScript**.

```
frontend/
├── app/
│   ├── layout.tsx              # Root layout
│   ├── page.tsx                # Home page
│   ├── dashboard/
│   │   └── page.tsx            # Dashboard page
│   ├── results/
│   │   └── page.tsx            # Results page
│   └── api/                    # API routes (optional)
├── components/
│   ├── ui/                     # shadcn/ui components
│   ├── system-status.tsx       # Backend status monitoring
│   ├── realtime-status.tsx     # WebSocket real-time status
│   ├── batch-prediction.tsx    # Batch file processing
│   ├── api-tester.tsx          # API connection tester
│   └── ...                     # Other components
├── hooks/
│   ├── use-api.ts              # API interaction hooks
│   └── use-websocket.ts        # WebSocket hooks
├── lib/
│   ├── axios.ts                # Enhanced API client
│   └── utils.ts                # Utility functions
└── styles/
    └── globals.css             # Global styles
```

## 🚀 Quick Start

### Installation

```bash
cd frontend
pnpm install
# or
npm install
```

### Environment Setup

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NODE_ENV=development
NEXT_PUBLIC_ENABLE_DEBUG=true
```

### Running Development Server

```bash
pnpm dev
# or
npm run dev
```

Visit `http://localhost:3000`

## 📦 Key Dependencies

- **Next.js 14**: React framework with App Router
- **React 18**: UI library
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling
- **shadcn/ui**: UI component library
- **Axios**: HTTP client
- **Lucide React**: Icons
- **Recharts**: Data visualization

## 🎨 Components

### Core Components

#### SystemStatus
Displays backend health with polling (30s intervals).

```tsx
import { SystemStatus } from "@/components/system-status";

<SystemStatus />
```

**Features:**
- Backend status indicator
- API version display
- Model availability badges
- Auto-refresh every 30 seconds

#### RealtimeStatus
Real-time status updates via WebSocket.

```tsx
import { RealtimeStatus } from "@/components/realtime-status";

<RealtimeStatus />
```

**Features:**
- WebSocket connection status
- Live backend updates (5s intervals)
- Model status indicators
- Connection health monitoring

#### BatchPrediction
Upload and process multiple files.

```tsx
import { BatchPrediction } from "@/components/batch-prediction";

<BatchPrediction />
```

**Features:**
- Multi-file selection
- Progress tracking
- Individual file status
- Batch statistics
- Error handling per file

#### ApiTester
Test API connectivity and endpoints.

```tsx
import { ApiTester } from "@/components/api-tester";

<ApiTester />
```

**Features:**
- 5 automated tests
- Response time tracking
- Pass/fail indicators
- Detailed error messages

### UI Components (shadcn/ui)

All located in `components/ui/`:
- `button.tsx`: Button variants
- `card.tsx`: Card containers
- `badge.tsx`: Status badges
- `progress.tsx`: Progress bars
- `alert.tsx`: Alert messages
- And more...

## 🔌 API Integration

### Enhanced Axios Client

Located at `lib/axios.ts`:

```typescript
import { api } from "@/lib/axios";

// Health check
const health = await api.health.check();

// Models info
const models = await api.models.info();

// Make prediction
const result = await api.predictions.predict(formData);

// List analyses
const analyses = await api.analyses.list();
```

**Features:**
- Request/response interceptors
- Automatic error handling
- Request logging (development)
- Timeout configuration (120s)
- Typed API methods

### Custom Hooks

#### useApi
Generic API hook with loading/error states.

```typescript
import { useApi } from "@/hooks/use-api";

const { data, loading, error, execute } = useApi<ResponseType>();

// Call API
await execute(async () => {
  return await api.health.check();
});
```

#### useHealthCheck
Specialized hook for health checks.

```typescript
import { useHealthCheck } from "@/hooks/use-api";

const { data, loading, error, refresh } = useHealthCheck();
```

#### useWebSocket
Generic WebSocket hook.

```typescript
import { useWebSocket } from "@/hooks/use-websocket";

const { isConnected, lastMessage, sendMessage } = useWebSocket(
  "ws://localhost:8000/api/v1/ws/status",
  {
    onMessage: (msg) => console.log(msg),
    reconnect: true,
    reconnectInterval: 5000
  }
);
```

#### useStatusWebSocket
Pre-configured status WebSocket.

```typescript
import { useStatusWebSocket } from "@/hooks/use-websocket";

const { isConnected, status } = useStatusWebSocket();
```

## 🎯 Pages

### Home (`app/page.tsx`)
Landing page with project introduction.

### Dashboard (`app/dashboard/page.tsx`)
Main application interface with:
- System status monitoring
- Real-time WebSocket status
- File upload section
- Batch prediction
- API tester
- Recent analyses
- Statistics cards

### Results (`app/results/page.tsx`)
Detailed analysis results visualization.

## 🎨 Styling

### Tailwind CSS
Utility-first CSS framework with custom configuration.

**Theme Configuration** (`tailwind.config.js`):
- Custom colors
- Dark mode support
- Custom animations
- Responsive breakpoints

### Global Styles
Located at `app/globals.css`:
- CSS variables for theming
- Base styles
- Custom animations
- Dark mode styles

### Component Styling
Uses Tailwind classes with shadcn/ui patterns:

```tsx
<Card className="border-2 hover:shadow-lg transition-shadow">
  <CardHeader>
    <CardTitle>Title</CardTitle>
  </CardHeader>
  <CardContent>
    Content
  </CardContent>
</Card>
```

## 🔧 Configuration

### TypeScript (`tsconfig.json`)
- Strict mode enabled
- Path aliases (@/components, @/lib, etc.)
- JSX preservation
- Module resolution

### Next.js (`next.config.js`)
- React strict mode
- Image optimization
- Environment variables
- Custom webpack config

## 📱 Responsive Design

All components are responsive with breakpoints:
- `sm`: 640px
- `md`: 768px
- `lg`: 1024px
- `xl`: 1280px
- `2xl`: 1536px

Example usage:
```tsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {/* Content */}
</div>
```

## 🔄 State Management

Currently using React's built-in state management:
- `useState`: Component state
- `useEffect`: Side effects
- `useCallback`: Memoized callbacks
- Custom hooks: Shared logic

For global state (if needed):
- Consider: Zustand, Jotai, or React Context

## 📊 Data Flow

### Prediction Flow
1. User uploads file in `FileUploadSection`
2. Component creates FormData
3. Calls `api.predictions.predict()`
4. Backend processes request
5. Results displayed in UI

### Real-time Updates Flow
1. Component mounts
2. `useStatusWebSocket()` establishes connection
3. Receives updates every 5 seconds
4. Updates component state
5. UI re-renders automatically

### Batch Processing Flow
1. User selects multiple files
2. Files added to state array
3. User clicks "Start Batch"
4. Loop processes each file sequentially
5. Progress bar updates
6. Individual file status updates

## 🧪 Testing

### Manual Testing
1. Start backend: `python backend/main.py`
2. Start frontend: `pnpm dev`
3. Navigate to `http://localhost:3000/dashboard`
4. Test each component

### API Testing
Use the built-in ApiTester component:
- Health check test
- System info test
- Models info test
- Individual model status tests

## 🚦 Error Handling

### API Errors
Handled by axios interceptors:
- Network errors (ECONNABORTED)
- Timeout errors
- HTTP errors (4xx, 5xx)
- Validation errors

### Component Errors
Use try-catch blocks:

```typescript
try {
  const result = await api.predictions.predict(data);
  // Handle success
} catch (error: any) {
  console.error("Prediction failed:", error.message);
  // Handle error
}
```

### WebSocket Errors
Auto-reconnect on disconnect:
```typescript
useWebSocket(url, {
  onError: (error) => console.error("WS Error:", error),
  reconnect: true,
  reconnectInterval: 5000
});
```

## 🎨 Icons

Using **Lucide React**:

```tsx
import { Upload, CheckCircle, AlertCircle } from "lucide-react";

<Upload className="h-4 w-4" />
<CheckCircle className="h-4 w-4 text-green-500" />
<AlertCircle className="h-4 w-4 text-red-500" />
```

## 📈 Performance Optimization

### Best Practices
1. **Code Splitting**: Automatic with Next.js
2. **Lazy Loading**: Import components as needed
3. **Memoization**: Use `useMemo` and `useCallback`
4. **Image Optimization**: Use Next.js `<Image>` component
5. **API Caching**: Implement with SWR or React Query (planned)

### Current Optimizations
- Automatic code splitting
- Server-side rendering (SSR)
- Static site generation (SSG) where possible
- Optimized bundle size with tree-shaking

## 🔐 Security

### Current Measures
- Environment variables for sensitive data
- CORS configuration
- Input validation
- XSS protection (React default)

### Future Enhancements
- Authentication/Authorization
- CSRF protection
- Rate limiting on client
- Secure WebSocket (WSS)

## 🚀 Deployment

### Build for Production

```bash
pnpm build
pnpm start
```

### Environment Variables
Set in production:
```env
NEXT_PUBLIC_API_URL=https://api.yourdomain.com/api/v1
NODE_ENV=production
```

### Vercel Deployment
1. Push to GitHub
2. Import project in Vercel
3. Set environment variables
4. Deploy

### Docker (Coming Soon)

## 📝 Development Guidelines

### Component Creation
1. Create in `components/` directory
2. Use TypeScript
3. Export as named export
4. Add JSDoc comments
5. Use shadcn/ui components

### Hook Creation
1. Create in `hooks/` directory
2. Prefix with `use`
3. Add TypeScript types
4. Handle loading/error states
5. Document usage

### Styling Guidelines
1. Use Tailwind utility classes
2. Follow shadcn/ui patterns
3. Maintain responsive design
4. Support dark mode
5. Use CSS variables for theme

## 🐛 Common Issues

### WebSocket Connection Failed
- Ensure backend is running
- Check NEXT_PUBLIC_API_URL
- Verify WebSocket endpoint is available

### CORS Errors
- Check backend CORS configuration
- Ensure origin is allowed
- Verify API URL is correct

### Hydration Errors
- Avoid using browser-only APIs in SSR
- Use `"use client"` directive when needed
- Check for mismatched HTML

## 📚 Resources

- [Next.js Docs](https://nextjs.org/docs)
- [React Docs](https://react.dev)
- [Tailwind CSS](https://tailwindcss.com)
- [shadcn/ui](https://ui.shadcn.com)
- [TypeScript](https://www.typescriptlang.org/docs)

## 🤝 Contributing

See main [CONTRIBUTING.md](../CONTRIBUTING.md).

## 📄 License

MIT License - See [LICENSE](../LICENSE).

**Forked from:** [hackbio-ca/hackathon](https://github.com/hackbio-ca/hackathon)
