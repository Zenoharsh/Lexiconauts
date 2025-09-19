import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";

// ---------------- Base API Setup ----------------
const API = axios.create({
  baseURL: "http://10.156.51.24:3000/api", // backend port
});

// Token automatically attach karne ke liye
API.interceptors.request.use(async (config) => {
  const token = await AsyncStorage.getItem("token");
  if (token) {
    // TS safe: headers ko merge karke assign kar rahe hain
    config.headers = {
      ...(config.headers as any), // error-free
      Authorization: `Bearer ${token}`,
    };
  }
  return config;
});

// ---------------- Response Types ----------------
export interface AuthResponse {
  token: string;
  user: {
    id: string;
    name: string;
    email: string;
  };
}

// ---------------- API Functions ----------------

// Auth
export const login = (email: string, password: string) =>
  API.post<AuthResponse>("/auth/login", { email, password });

export const signup = (email: string, password: string, name: string) =>
  API.post<AuthResponse>("/auth/signup", { email, password, name });

// Video upload
export const uploadVideo = (formData: FormData) =>
  API.post("/videos/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

// Protected route example
export const getProtectedData = () => API.get("/protected/data");
