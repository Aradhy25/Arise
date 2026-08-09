import { Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./lib/auth.jsx";
import LandingPage from "./pages/LandingPage.jsx";
import ScanPage from "./pages/ScanPage.jsx";
import AuthPage from "./pages/AuthPage.jsx";
import DashboardPage from "./pages/DashboardPage.jsx";
import LivePage from "./pages/LivePage.jsx";

function Protected({ children }) {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) return <Navigate to="/auth" replace />;
  return children;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/scan" element={<ScanPage />} />
      <Route path="/auth" element={<AuthPage />} />
      <Route
        path="/app"
        element={
          <Protected>
            <DashboardPage />
          </Protected>
        }
      />
      <Route path="/live" element={<LivePage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
