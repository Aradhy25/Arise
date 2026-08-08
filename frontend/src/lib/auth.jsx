import { createContext, useContext, useMemo, useState } from "react";
import { api } from "./api";

const AuthContext = createContext(null);
const STORAGE_KEY = "deepguard_auth";

function loadStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }) {
  const [auth, setAuth] = useState(loadStored);

  function persist(next) {
    setAuth(next);
    if (next) localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    else localStorage.removeItem(STORAGE_KEY);
  }

  async function register(payload) {
    const data = await api("/api/auth/register", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    persist({ token: data.access_token, user: data.user });
    return data;
  }

  async function login(payload) {
    const data = await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    persist({ token: data.access_token, user: data.user });
    return data;
  }

  function logout() {
    persist(null);
  }

  const value = useMemo(
    () => ({
      user: auth?.user ?? null,
      token: auth?.token ?? null,
      isAuthenticated: Boolean(auth?.token),
      register,
      login,
      logout,
    }),
    [auth]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
