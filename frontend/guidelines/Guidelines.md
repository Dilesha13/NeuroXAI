# NeuroXAI Frontend

NeuroXAI Frontend is a React and TypeScript web application for EEG-based neonatal seizure analysis. It provides the user interface for authentication, EEG upload, patient record viewing, AI analysis, explainability, report generation, and account settings.

This project is built with Vite and uses a component-based UI structure for a responsive clinical dashboard experience.

## Features

- Landing page for product introduction
- User authentication flow
  - Sign in
  - Sign up
  - Email verification
- Protected dashboard area
- EEG upload page with EDF file validation
- Patient records view
- AI analysis page for seizure detection results
- Explainability page for model insights
- Reports page for generated reports
- Settings page for profile, preferences, and security options
- Theme support

## Tech Stack

- React
- TypeScript
- Vite
- Tailwind CSS
- React Router
- Recharts
- Radix UI components
- Lucide React

## Project Structure

```text
frontend/
├── guidelines/
│   └── Guidelines.md
├── public/
│   └── images/
├── src/
│   ├── app/
│   │   ├── components/
│   │   │   ├── auth/
│   │   │   └── ui/
│   │   ├── context/
│   │   ├── layouts/
│   │   ├── lib/
│   │   ├── pages/
│   │   ├── App.tsx
│   │   └── routes.tsx
│   ├── assets/
│   ├── styles/
│   ├── config.ts
│   └── main.tsx
├── index.html
├── package.json
├── package-lock.json
├── postcss.config.mjs
└── vite.config.js
```

## Main Pages

### Public Pages

- `/` - Landing page
- `/signin` - Sign in page
- `/signup` - Sign up page
- `/verify-email` - Email verification page

### Protected Pages

- `/app` - Dashboard
- `/app/upload` - Upload EEG
- `/app/patients` - Patient records
- `/app/analysis` - AI analysis
- `/app/explainability` - Explainability
- `/app/reports` - Reports
- `/app/settings` - Settings

## API Configuration

The frontend currently points to the backend API through `src/config.ts`.

```ts
export const API_BASE_URL = "http://127.0.0.1:8000/api/v1";
```

Some pages also contain the same backend URL directly in the source. Before deployment, update these URLs if your backend is hosted elsewhere.

## Prerequisites

Make sure the following are installed:

- Node.js 18 or later
- npm 9 or later

## Installation

```bash
npm install
```

## Running the Project

Start the development server:

```bash
npm run dev
```

The app will usually be available at:

```text
http://localhost:5173
```

## Production Build

Create a production build:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

## Authentication

The application stores authentication data in local storage after login.

Typical keys used:

- `auth_token`
- `auth_user`
- `latestInferenceResult`
- `neuroxai-user`

Protected pages depend on valid backend authentication responses.

## EEG Upload Workflow

The Upload EEG page supports EDF file selection and drag-and-drop upload. The implemented flow includes:

1. Select or drop an EDF file
2. Enter patient ID and recording date
3. Create or find the patient record through the backend
4. Upload the EEG file
5. Trigger backend inference
6. Save the returned result in local storage
7. Navigate to analysis and explainability pages

## Notes for Deployment

Before deployment, review these points:

- Replace hardcoded localhost API URLs with your deployed backend URL
- Confirm CORS settings on the backend allow your frontend domain
- Ensure authentication endpoints are reachable
- Make sure report and profile image URLs also use the correct backend base URL

## Useful Scripts

```bash
npm run dev
npm run build
npm run preview
```

## Known Considerations

- The project mixes `src/config.ts` and hardcoded backend URLs in some files
- Backend availability is required for login, settings, upload, analysis, and reports
- Local storage is used to pass inference results between pages

## Suggested Improvements

- Centralize all API URLs in one config file
- Add environment variable support
- Add form validation utilities
- Add loading and error boundaries
- Add automated tests
- Remove unrelated files from the UI component folder

## License

This project is for academic and research purposes unless you define a separate license.
