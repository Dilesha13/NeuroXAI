import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { useTheme } from '../context/ThemeContext';
import {
  Brain,
  LayoutDashboard,
  Upload,
  Users,
  Activity,
  Brain as BrainIcon,
  FileText,
  Settings,
  LogOut,
  Menu,
  X,
  Moon,
  Sun,
  Bell,
  User,
} from 'lucide-react';
import { clearAuth, fetchCurrentUser, getStoredUser, type AuthUser } from '../lib/auth';

const navigation = [
  { name: 'Dashboard', href: '/app', icon: LayoutDashboard },
  { name: 'Upload EEG', href: '/app/upload', icon: Upload },
  { name: 'Patient Records', href: '/app/patients', icon: Users },
  { name: 'AI Analysis', href: '/app/analysis', icon: Activity },
  { name: 'Explainability', href: '/app/explainability', icon: BrainIcon },
  { name: 'Reports', href: '/app/reports', icon: FileText },
  { name: 'Settings', href: '/app/settings', icon: Settings },
];

export default function DashboardLayout() {
  const { theme, toggleTheme } = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(getStoredUser());

  useEffect(() => {
    const syncUser = async () => {
      try {
        const user = await fetchCurrentUser();
        setCurrentUser(user);
      } catch {
        clearAuth();
        navigate('/signin');
      }
    };

    syncUser();
  }, [navigate]);

  useEffect(() => {
    const handleStorage = () => {
      setCurrentUser(getStoredUser());
    };

    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const handleLogout = () => {
    clearAuth();
    navigate('/signin');
  };

  const displayName = currentUser?.full_name || 'User';
  const displayRole = currentUser?.specialization || 'Medical Professional';

  const profilePhotoUrl = currentUser?.profile_photo_url
    ? `http://127.0.0.1:8000${currentUser.profile_photo_url}`
    : null;

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 dark:bg-slate-950 dark:text-slate-100">
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 transform border-r border-gray-200 bg-white transition-transform duration-200 ease-in-out dark:border-slate-700 dark:bg-slate-800 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } lg:translate-x-0`}
      >
        <div className="flex h-full flex-col">
          <div className="flex h-[72px] items-center justify-between border-b border-gray-200 px-6 dark:border-slate-700">
            <Link to="/" className="flex items-center gap-3">
              <Brain className="h-8 w-8 text-blue-600 dark:text-cyan-400" />
              <span className="text-[2rem] font-bold leading-none text-slate-900 dark:text-white">
                NeuroXAI
              </span>
            </Link>

            <button
              onClick={() => setSidebarOpen(false)}
              className="rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:text-slate-400 dark:hover:bg-slate-700 lg:hidden"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <nav className="flex-1 space-y-2 overflow-y-auto px-4 py-6">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;

              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={`flex items-center gap-3 rounded-xl px-4 py-3 text-[17px] font-medium transition-all ${
                    isActive
                      ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/30 dark:text-cyan-400'
                      : 'text-gray-700 hover:bg-gray-100 dark:text-slate-200 dark:hover:bg-slate-700/70'
                  }`}
                >
                  <item.icon className="h-5 w-5 shrink-0" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>

          <div className="border-t border-gray-200 p-4 dark:border-slate-700">
            <button
              onClick={handleLogout}
              className="flex w-full items-center gap-3 rounded-xl px-4 py-3 text-[17px] font-medium text-gray-700 transition-colors hover:bg-gray-100 dark:text-slate-200 dark:hover:bg-slate-700/70"
            >
              <LogOut className="h-5 w-5" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="lg:pl-64">
        <header className="sticky top-0 z-30 border-b border-gray-200 bg-white dark:border-slate-700 dark:bg-slate-800">
          <div className="flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setSidebarOpen(true)}
                className="rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:text-slate-400 dark:hover:bg-slate-700 lg:hidden"
              >
                <Menu className="h-6 w-6" />
              </button>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={toggleTheme}
                className="rounded-lg p-2 text-gray-500 transition-colors hover:bg-gray-100 dark:text-slate-400 dark:hover:bg-slate-700"
              >
                {theme === 'light' ? (
                  <Moon className="h-5 w-5" />
                ) : (
                  <Sun className="h-5 w-5" />
                )}
              </button>

              <button className="relative rounded-lg p-2 text-gray-500 transition-colors hover:bg-gray-100 dark:text-slate-400 dark:hover:bg-slate-700">
                <Bell className="h-5 w-5" />
                <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-red-500" />
              </button>

              <Link
                to="/app/settings"
                className="flex items-center gap-3 rounded-lg px-2 py-1 transition-colors hover:bg-gray-100 dark:hover:bg-slate-700"
              >
                <div className="flex h-9 w-9 items-center justify-center overflow-hidden rounded-full bg-blue-100 dark:bg-blue-900/50">
                  {profilePhotoUrl ? (
                    <img
                      src={profilePhotoUrl}
                      alt="Profile"
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <User className="h-4 w-4 text-blue-600 dark:text-blue-300" />
                  )}
                </div>

                <div className="hidden text-left sm:block">
                  <p className="text-sm font-semibold leading-tight text-slate-900 dark:text-white">
                    {displayName}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-slate-400">
                    {displayRole}
                  </p>
                </div>
              </Link>
            </div>
          </div>
        </header>

        <main className="px-4 py-6 sm:px-6 lg:px-7">
          <Outlet />
        </main>
      </div>
    </div>
  );
}