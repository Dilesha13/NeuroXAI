import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { API_BASE_URL } from '../lib/auth';

export default function VerifyEmailPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState('Verifying your email...');

  useEffect(() => {
    const token = searchParams.get('token');

    if (!token) {
      setStatus('error');
      setMessage('Verification token is missing.');
      return;
    }

    const verifyEmail = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/auth/verify-email`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ token }),
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Email verification failed');
        }

        setStatus('success');
        setMessage(data.message || 'Email verified successfully.');

        setTimeout(() => {
          navigate('/signin');
        }, 2000);
      } catch (err) {
        setStatus('error');
        setMessage(err instanceof Error ? err.message : 'Email verification failed');
      }
    };

    verifyEmail();
  }, [searchParams, navigate]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-900 px-4">
      <div className="w-full max-w-md rounded-2xl bg-white dark:bg-slate-800 shadow-xl p-8 text-center">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">
          Email Verification
        </h1>

        <p
          className={`text-sm ${
            status === 'success'
              ? 'text-green-600 dark:text-green-400'
              : status === 'error'
              ? 'text-red-600 dark:text-red-400'
              : 'text-slate-600 dark:text-slate-300'
          }`}
        >
          {message}
        </p>

        {status === 'success' && (
          <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">
            Redirecting to sign in...
          </p>
        )}

        {status === 'error' && (
          <div className="mt-6">
            <Link
              to="/signin"
              className="text-blue-600 dark:text-teal-400 font-semibold hover:underline"
            >
              Go to Sign In
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}