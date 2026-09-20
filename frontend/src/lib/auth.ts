import { create } from "zustand";
import { apiFetch } from "./api";

export interface User {
  id: string;
  email: string;
  name: string;
  expertise_level: string;
  preferred_language: string;
  created_at: string;
}

interface AuthState {
  user: User | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (
    email: string,
    password: string,
    name: string,
    expertise_level: string,
  ) => Promise<void>;
  logout: () => void;
  fetchMe: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isLoading: true,

  login: async (email, password) => {
    const data = await apiFetch<{
      access_token: string;
      refresh_token: string;
      user: User;
    }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    localStorage.setItem("access_token", data.access_token);
    localStorage.setItem("refresh_token", data.refresh_token);
    set({ user: data.user });
  },

  register: async (email, password, name, expertise_level) => {
    const data = await apiFetch<{
      access_token: string;
      refresh_token: string;
      user: User;
    }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, name, expertise_level }),
    });
    localStorage.setItem("access_token", data.access_token);
    localStorage.setItem("refresh_token", data.refresh_token);
    set({ user: data.user });
  },

  logout: () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    set({ user: null });
  },

  fetchMe: async () => {
    try {
      const user = await apiFetch<User>("/auth/me");
      set({ user, isLoading: false });
    } catch {
      set({ user: null, isLoading: false });
    }
  },
}));
