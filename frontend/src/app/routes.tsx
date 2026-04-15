import { createBrowserRouter } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import SignInPage from "./pages/SignInPage";
import SignUpPage from "./pages/SignUpPage";
import VerifyEmailPage from "./pages/VerifyEmailPage";
import DashboardLayout from "./layouts/DashboardLayout";
import Dashboard from "./pages/Dashboard";
import UploadEEG from "./pages/UploadEEG";
import PatientRecords from "./pages/PatientRecords";
import AIAnalysis from "./pages/AIAnalysis";
import Explainability from "./pages/Explainability";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <LandingPage />,
  },
  {
    path: "/signin",
    element: <SignInPage />,
  },
  {
    path: "/signup",
    element: <SignUpPage />,
  },
  {
    path: "/verify-email",
    element: <VerifyEmailPage />,
  },
  {
    path: "/app",
    element: <DashboardLayout />,
    children: [
      {
        index: true,
        element: <Dashboard />,
      },
      {
        path: "upload",
        element: <UploadEEG />,
      },
      {
        path: "patients",
        element: <PatientRecords />,
      },
      {
        path: "analysis",
        element: <AIAnalysis />,
      },
      {
        path: "explainability",
        element: <Explainability />,
      },
      {
        path: "reports",
        element: <Reports />,
      },
      {
        path: "settings",
        element: <Settings />,
      },
    ],
  },
]);