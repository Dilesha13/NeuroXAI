import { Link, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { useTheme } from '../context/ThemeContext';
import { Moon, Sun, Brain } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { saveAuth, signUpRequest } from '../lib/auth';

export default function SignUpPage() {
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    password: '',
    confirmPassword: '',
    hospital: '',
  });

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (formData.password !== formData.confirmPassword) {
      setError('Password and confirm password do not match');
      return;
    }

    try {
      setIsLoading(true);

      const data = await signUpRequest({
        full_name: formData.fullName,
        email: formData.email,
        password: formData.password,
        hospital: formData.hospital,
        specialization: 'Neurologist',
      });

      alert(data.message || 'Please verify your email before signing in');

      navigate('/signin');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to create account');
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 flex items-center justify-center p-4">
      <button
        onClick={toggleTheme}
        className="fixed top-4 right-4 p-2 rounded-lg bg-white dark:bg-gray-800 shadow-lg hover:shadow-xl transition-all"
      >
        {theme === 'light' ? (
          <Moon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
        ) : (
          <Sun className="h-5 w-5 text-gray-600 dark:text-gray-400" />
        )}
      </button>

      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-2 mb-2">
            <Brain className="h-10 w-10 text-blue-600 dark:text-teal-400" />
            <span className="text-2xl font-bold text-gray-900 dark:text-white">NeuroXAI</span>
          </Link>
          <p className="text-gray-600 dark:text-gray-400">
            AI-Powered Neonatal Seizure Detection
          </p>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-2xl shadow-2xl p-8 border border-gray-100 dark:border-gray-800">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Create NeuroXAI Account
          </h1>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <Label htmlFor="fullName" className="text-gray-700 dark:text-gray-300">
                Full Name
              </Label>
              <Input
                id="fullName"
                name="fullName"
                type="text"
                value={formData.fullName}
                onChange={handleChange}
                placeholder="Dr. Jane Smith"
                className="mt-1 dark:bg-slate-800 dark:border-gray-700"
                required
              />
            </div>

            <div>
              <Label htmlFor="email" className="text-gray-700 dark:text-gray-300">
                Email
              </Label>
              <Input
                id="email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="doctor@hospital.com"
                className="mt-1 dark:bg-slate-800 dark:border-gray-700"
                required
              />
            </div>

            <div>
              <Label htmlFor="password" className="text-gray-700 dark:text-gray-300">
                Password
              </Label>
              <Input
                id="password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="••••••••"
                className="mt-1 dark:bg-slate-800 dark:border-gray-700"
                required
              />
            </div>

            <div>
              <Label htmlFor="confirmPassword" className="text-gray-700 dark:text-gray-300">
                Confirm Password
              </Label>
              <Input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                value={formData.confirmPassword}
                onChange={handleChange}
                placeholder="••••••••"
                className="mt-1 dark:bg-slate-800 dark:border-gray-700"
                required
              />
            </div>

            <div>
              <Label htmlFor="hospital" className="text-gray-700 dark:text-gray-300">
                Hospital / Organization
              </Label>
              <Input
                id="hospital"
                name="hospital"
                type="text"
                value={formData.hospital}
                onChange={handleChange}
                placeholder="City General Hospital"
                className="mt-1 dark:bg-slate-800 dark:border-gray-700"
                required
              />
            </div>

            {error && (
              <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-600 dark:border-red-900/40 dark:bg-red-950/30 dark:text-red-300">
                {error}
              </div>
            )}

            <Button
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 hover:bg-blue-700 dark:bg-teal-600 dark:hover:bg-teal-700 text-white"
            >
              {isLoading ? 'Creating Account...' : 'Create Account'}
            </Button>
          </form>

          <p className="mt-6 text-center text-sm text-gray-600 dark:text-gray-400">
            Already have an account?{' '}
            <Link
              to="/signin"
              className="text-blue-600 dark:text-teal-400 font-semibold hover:underline"
            >
              Sign In
            </Link>
          </p>
        </div>

        <p className="mt-6 text-xs text-center text-gray-500 dark:text-gray-500">
          By creating an account, you agree to comply with medical data protection regulations and use this system for authorized clinical purposes only.
        </p>
      </div>
    </div>
  );
}