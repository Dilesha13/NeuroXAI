export const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

export type AuthUser = {
  id: number;
  full_name: string;
  email: string;
  hospital?: string | null;
  specialization?: string | null;
  profile_photo_url?: string | null;
  is_active?: boolean;
  created_at?: string;
};

export type MessageResponse = {
  message: string;
};

type AuthResponse = {
  access_token: string;
  token_type: string;
  user: AuthUser;
};

export function saveAuth(data: AuthResponse) {
  localStorage.setItem('neuroxai-token', data.access_token);
  localStorage.setItem('neuroxai-user', JSON.stringify(data.user));
}

export function getToken() {
  return localStorage.getItem('neuroxai-token');
}

export function getStoredUser(): AuthUser | null {
  const raw = localStorage.getItem('neuroxai-user');
  if (!raw) return null;

  try {
    return JSON.parse(raw) as AuthUser;
  } catch {
    return null;
  }
}

export function clearAuth() {
  localStorage.removeItem('neuroxai-token');
  localStorage.removeItem('neuroxai-user');
}

export async function signInRequest(email: string, password: string) {
  const response = await fetch(`${API_BASE_URL}/auth/signin`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      email,
      password,
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || 'Sign in failed');
  }

  return data as AuthResponse;
}

export async function signUpRequest(payload: {
  full_name: string;
  email: string;
  password: string;
  hospital?: string;
  specialization?: string;
}) {
  const response = await fetch(`${API_BASE_URL}/auth/signup`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || 'Sign up failed');
  }

  return data as MessageResponse;
}

export async function fetchCurrentUser() {
  const token = getToken();
  if (!token) {
    throw new Error('No auth token found');
  }

  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || 'Failed to load current user');
  }

  localStorage.setItem('neuroxai-user', JSON.stringify(data));
  return data as AuthUser;
}