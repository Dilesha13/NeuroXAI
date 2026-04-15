import { useEffect, useState } from 'react';
import {
  User,
  Lock,
  Bell,
  Palette,
  FileDown,
  Save,
  ShieldAlert,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Button } from '../components/ui/button';
import { Switch } from '../components/ui/switch';
import { Separator } from '../components/ui/separator';
import { useTheme } from '../context/ThemeContext';
import { API_BASE_URL, getToken, getStoredUser, fetchCurrentUser } from '../lib/auth';

type ProfileState = {
  fullName: string;
  email: string;
  hospital: string;
  specialization: string;
};

type NotificationsState = {
  emailAlerts: boolean;
  seizureDetection: boolean;
  weeklyReports: boolean;
  systemUpdates: boolean;
};

type SecurityState = {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
};

type AppSettingsState = {
  exportFormat: string;
  dataRetention: string;
};

export default function Settings() {
  const { theme, toggleTheme } = useTheme();

  const [profile, setProfile] = useState<ProfileState>({
    fullName: '',
    email: '',
    hospital: '',
    specialization: '',
  });

  const [notifications, setNotifications] = useState<NotificationsState>({
    emailAlerts: true,
    seizureDetection: true,
    weeklyReports: false,
    systemUpdates: true,
  });

  const [security, setSecurity] = useState<SecurityState>({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });

  const [appSettings, setAppSettings] = useState<AppSettingsState>({
    exportFormat: 'PDF',
    dataRetention: '3 months',
  });

  const [profilePhotoUrl, setProfilePhotoUrl] = useState<string | null>(null);
  const [isUploadingPhoto, setIsUploadingPhoto] = useState(false);

  const [isLoading, setIsLoading] = useState(true);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [isSavingPreferences, setIsSavingPreferences] = useState(false);
  const [isUpdatingPassword, setIsUpdatingPassword] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const token = getToken();

  const handleProfileChange = (field: keyof ProfileState, value: string) => {
    setProfile((prev) => ({ ...prev, [field]: value }));
  };

  const handleNotificationChange = (field: keyof NotificationsState, value: boolean) => {
    setNotifications((prev) => ({ ...prev, [field]: value }));
  };

  const handleSecurityChange = (field: keyof SecurityState, value: string) => {
    setSecurity((prev) => ({ ...prev, [field]: value }));
  };

  const handleAppSettingChange = (field: keyof AppSettingsState, value: string) => {
    setAppSettings((prev) => ({ ...prev, [field]: value }));
  };

  const updateStoredUser = (updated: {
    full_name: string;
    email: string;
    hospital?: string | null;
    specialization?: string | null;
    profile_photo_url?: string | null;
  }) => {
    const existing = getStoredUser();
    if (!existing) return;

    const merged = {
      ...existing,
      full_name: updated.full_name,
      email: updated.email,
      hospital: updated.hospital ?? '',
      specialization: updated.specialization ?? '',
      profile_photo_url: updated.profile_photo_url ?? existing.profile_photo_url ?? null,
    };

    localStorage.setItem('neuroxai-user', JSON.stringify(merged));
    window.dispatchEvent(new Event('storage'));
  };

  const loadSettings = async () => {
    if (!token) {
      setError('You are not signed in');
      setIsLoading(false);
      return;
    }

    try {
      setError('');
      setMessage('');

      const [profileRes, preferencesRes] = await Promise.all([
        fetch(`${API_BASE_URL}/settings/profile`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }),
        fetch(`${API_BASE_URL}/settings/preferences`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }),
      ]);

      const profileData = await profileRes.json();
      const preferencesData = await preferencesRes.json();

      if (!profileRes.ok) {
        throw new Error(profileData.detail || 'Failed to load profile');
      }

      if (!preferencesRes.ok) {
        throw new Error(preferencesData.detail || 'Failed to load preferences');
      }

      setProfile({
        fullName: profileData.full_name || '',
        email: profileData.email || '',
        hospital: profileData.hospital || '',
        specialization: profileData.specialization || '',
      });

      setProfilePhotoUrl(
        profileData.profile_photo_url
          ? `http://127.0.0.1:8000${profileData.profile_photo_url}`
          : null
      );

      setNotifications({
        emailAlerts: preferencesData.email_alerts,
        seizureDetection: preferencesData.seizure_detection_alerts,
        weeklyReports: preferencesData.weekly_reports,
        systemUpdates: preferencesData.system_updates,
      });

      setAppSettings({
        exportFormat: preferencesData.export_format || 'PDF',
        dataRetention: preferencesData.data_retention || '3 months',
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load settings');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadSettings();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSaveProfile = async () => {
    if (!token) {
      setError('You are not signed in');
      return;
    }

    try {
      setIsSavingProfile(true);
      setError('');
      setMessage('');

      const response = await fetch(`${API_BASE_URL}/settings/profile`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          full_name: profile.fullName,
          email: profile.email,
          hospital: profile.hospital,
          specialization: profile.specialization,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to save profile');
      }

      updateStoredUser(data);
      if (data.profile_photo_url) {
        setProfilePhotoUrl(`http://127.0.0.1:8000${data.profile_photo_url}`);
      }
      setMessage('Profile updated successfully');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save profile');
    } finally {
      setIsSavingProfile(false);
    }
  };

  const handlePhotoUpload = async (file: File) => {
    if (!token) {
      setError('You are not signed in');
      return;
    }

    try {
      setIsUploadingPhoto(true);
      setError('');
      setMessage('');

      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/settings/profile/photo`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to upload profile photo');
      }

      const fullPhotoUrl = `http://127.0.0.1:8000${data.profile_photo_url}`;
      setProfilePhotoUrl(fullPhotoUrl);

      const refreshedUser = await fetchCurrentUser();
      localStorage.setItem('neuroxai-user', JSON.stringify(refreshedUser));
      window.dispatchEvent(new Event('storage'));

      setMessage('Profile photo uploaded successfully');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload profile photo');
    } finally {
      setIsUploadingPhoto(false);
    }
  };

  const handleSavePreferences = async () => {
    if (!token) {
      setError('You are not signed in');
      return;
    }

    try {
      setIsSavingPreferences(true);
      setError('');
      setMessage('');

      const response = await fetch(`${API_BASE_URL}/settings/preferences`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          email_alerts: notifications.emailAlerts,
          seizure_detection_alerts: notifications.seizureDetection,
          weekly_reports: notifications.weeklyReports,
          system_updates: notifications.systemUpdates,
          export_format: appSettings.exportFormat,
          data_retention: appSettings.dataRetention,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to save preferences');
      }

      setMessage('Preferences updated successfully');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save preferences');
    } finally {
      setIsSavingPreferences(false);
    }
  };

  const handleUpdatePassword = async () => {
    if (!token) {
      setError('You are not signed in');
      return;
    }

    try {
      setIsUpdatingPassword(true);
      setError('');
      setMessage('');

      const response = await fetch(`${API_BASE_URL}/settings/password`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          current_password: security.currentPassword,
          new_password: security.newPassword,
          confirm_password: security.confirmPassword,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to update password');
      }

      setMessage('Password updated successfully');
      setSecurity({
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update password');
    } finally {
      setIsUpdatingPassword(false);
    }
  };

  const handleDeleteAccount = async () => {
    if (!token) {
      setError('You are not signed in');
      return;
    }

    const confirmed = window.confirm(
      'Are you sure you want to delete your account? This action cannot be undone.'
    );

    if (!confirmed) return;

    try {
      setError('');
      setMessage('');

      const response = await fetch(`${API_BASE_URL}/settings/account`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to delete account');
      }

      localStorage.removeItem('neuroxai-token');
      localStorage.removeItem('neuroxai-user');
      window.location.href = '/signin';
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete account');
    }
  };

  const sectionRowClass =
    'flex flex-col gap-4 md:flex-row md:items-center md:justify-between';
  const sectionTextWrapClass = 'min-w-0';
  const sectionTitleClass = 'text-base font-semibold text-slate-900 dark:text-white';
  const sectionDescClass = 'text-sm text-slate-600 dark:text-slate-400';
  const inputClass =
    'mt-2 h-12 rounded-lg border-slate-200 bg-white text-slate-900 placeholder:text-slate-400 dark:border-slate-700 dark:bg-slate-950 dark:text-white';
  const selectClass =
    'h-12 min-w-[120px] rounded-xl border border-slate-200 bg-white px-4 text-sm font-medium text-slate-900 outline-none transition dark:border-slate-700 dark:bg-slate-950 dark:text-white';

  if (isLoading) {
    return (
      <div className="space-y-6 pb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            Settings
          </h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
            Loading settings...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
          Settings
        </h1>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
          Manage your account and application preferences
        </p>
      </div>

      {message && (
        <div className="rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700 dark:border-green-900/40 dark:bg-green-950/30 dark:text-green-300">
          {message}
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/40 dark:bg-red-950/30 dark:text-red-300">
          {error}
        </div>
      )}

      {/* User Profile */}
      <Card className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-800">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg font-semibold text-slate-900 dark:text-white">
            <User className="h-5 w-5 text-cyan-500" />
            User Profile
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
            <div className="flex h-24 w-24 items-center justify-center overflow-hidden rounded-full bg-blue-700">
              {profilePhotoUrl ? (
                <img
                  src={profilePhotoUrl}
                  alt="Profile"
                  className="h-full w-full object-cover"
                />
              ) : (
                <User className="h-12 w-12 text-blue-200" />
              )}
            </div>

            <div>
              <label className="inline-flex cursor-pointer items-center rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-900 dark:border-slate-600 dark:text-white">
                {isUploadingPhoto ? 'Uploading...' : 'Change Photo'}
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      handlePhotoUpload(file);
                    }
                  }}
                  disabled={isUploadingPhoto}
                />
              </label>
              <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
                JPG, PNG, GIF or WEBP. Max size 2MB
              </p>
            </div>
          </div>

          <Separator className="bg-slate-200 dark:bg-slate-700" />

          <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
            <div>
              <Label htmlFor="fullName" className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Full Name
              </Label>
              <Input
                id="fullName"
                type="text"
                value={profile.fullName}
                onChange={(e) => handleProfileChange('fullName', e.target.value)}
                className={inputClass}
              />
            </div>

            <div>
              <Label htmlFor="email" className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Email Address
              </Label>
              <Input
                id="email"
                type="email"
                value={profile.email}
                onChange={(e) => handleProfileChange('email', e.target.value)}
                className={inputClass}
              />
            </div>

            <div>
              <Label htmlFor="hospital" className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Hospital / Organization
              </Label>
              <Input
                id="hospital"
                type="text"
                value={profile.hospital}
                onChange={(e) => handleProfileChange('hospital', e.target.value)}
                className={inputClass}
              />
            </div>

            <div>
              <Label htmlFor="specialization" className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Specialization
              </Label>
              <Input
                id="specialization"
                type="text"
                value={profile.specialization}
                onChange={(e) => handleProfileChange('specialization', e.target.value)}
                className={inputClass}
              />
            </div>
          </div>

          <div className="flex justify-end">
            <Button
              onClick={handleSaveProfile}
              disabled={isSavingProfile}
              className="h-11 rounded-lg bg-teal-500 px-5 text-white hover:bg-teal-600 dark:bg-teal-500 dark:hover:bg-teal-600"
            >
              <Save className="mr-2 h-4 w-4" />
              {isSavingProfile ? 'Saving...' : 'Save Changes'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Security */}
      <Card className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-800">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg font-semibold text-slate-900 dark:text-white">
            <Lock className="h-5 w-5 text-violet-400" />
            Security
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-5">
          <div>
            <Label htmlFor="currentPassword" className="text-sm font-medium text-slate-700 dark:text-slate-300">
              Current Password
            </Label>
            <Input
              id="currentPassword"
              type="password"
              value={security.currentPassword}
              onChange={(e) => handleSecurityChange('currentPassword', e.target.value)}
              className={inputClass}
            />
          </div>

          <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
            <div>
              <Label htmlFor="newPassword" className="text-sm font-medium text-slate-700 dark:text-slate-300">
                New Password
              </Label>
              <Input
                id="newPassword"
                type="password"
                value={security.newPassword}
                onChange={(e) => handleSecurityChange('newPassword', e.target.value)}
                className={inputClass}
              />
            </div>

            <div>
              <Label htmlFor="confirmPassword" className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Confirm Password
              </Label>
              <Input
                id="confirmPassword"
                type="password"
                value={security.confirmPassword}
                onChange={(e) => handleSecurityChange('confirmPassword', e.target.value)}
                className={inputClass}
              />
            </div>
          </div>

          <div className="flex justify-end">
            <Button
              onClick={handleUpdatePassword}
              disabled={isUpdatingPassword}
              className="h-11 rounded-lg bg-fuchsia-600 px-5 text-white hover:bg-fuchsia-700"
            >
              {isUpdatingPassword ? 'Updating...' : 'Update Password'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Notification Preferences
      <Card className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-800">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg font-semibold text-slate-900 dark:text-white">
            <Bell className="h-5 w-5 text-yellow-400" />
            Notification Preferences
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          <div className={sectionRowClass}>
            <div className={sectionTextWrapClass}>
              <p className={sectionTitleClass}>Email Alerts</p>
              <p className={sectionDescClass}>Receive important notifications via email</p>
            </div>
            <Switch
              checked={notifications.emailAlerts}
              onCheckedChange={(checked) => handleNotificationChange('emailAlerts', checked)}
            />
          </div>

          <Separator className="bg-slate-200 dark:bg-slate-700" />

          <div className={sectionRowClass}>
            <div className={sectionTextWrapClass}>
              <p className={sectionTitleClass}>Seizure Detection Alerts</p>
              <p className={sectionDescClass}>Get notified immediately when seizures are detected</p>
            </div>
            <Switch
              checked={notifications.seizureDetection}
              onCheckedChange={(checked) => handleNotificationChange('seizureDetection', checked)}
            />
          </div>

          <Separator className="bg-slate-200 dark:bg-slate-700" />

          <div className={sectionRowClass}>
            <div className={sectionTextWrapClass}>
              <p className={sectionTitleClass}>Weekly Reports</p>
              <p className={sectionDescClass}>Receive weekly summary of analysis activities</p>
            </div>
            <Switch
              checked={notifications.weeklyReports}
              onCheckedChange={(checked) => handleNotificationChange('weeklyReports', checked)}
            />
          </div>

          <Separator className="bg-slate-200 dark:bg-slate-700" />

          <div className={sectionRowClass}>
            <div className={sectionTextWrapClass}>
              <p className={sectionTitleClass}>System Updates</p>
              <p className={sectionDescClass}>Get notified about new features and improvements</p>
            </div>
            <Switch
              checked={notifications.systemUpdates}
              onCheckedChange={(checked) => handleNotificationChange('systemUpdates', checked)}
            />
          </div>

          <div className="flex justify-end">
            <Button
              onClick={handleSavePreferences}
              disabled={isSavingPreferences}
              className="h-11 rounded-lg bg-teal-500 px-5 text-white hover:bg-teal-600 dark:bg-teal-500 dark:hover:bg-teal-600"
            >
              <Save className="mr-2 h-4 w-4" />
              {isSavingPreferences ? 'Saving...' : 'Save Preferences'}
            </Button>
          </div>
        </CardContent>
      </Card> */}

      {/* App Settings */}
      <Card className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-800">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg font-semibold text-slate-900 dark:text-white">
            <Palette className="h-5 w-5 text-cyan-500" />
            App Settings
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          <div className={sectionRowClass}>
            <div className={sectionTextWrapClass}>
              <p className={sectionTitleClass}>Theme Preference</p>
              <p className={sectionDescClass}>
                Current theme: {theme === 'light' ? 'Light Mode' : 'Dark Mode'}
              </p>
            </div>
            <Button
              variant="outline"
              onClick={toggleTheme}
              className="h-11 rounded-lg border-slate-300 px-4 dark:border-slate-600 dark:bg-transparent dark:text-white"
            >
              Switch to {theme === 'light' ? 'Dark' : 'Light'} Mode
            </Button>
          </div>

          <Separator className="bg-slate-200 dark:bg-slate-700" />

          <div className={sectionRowClass}>
            <div className={sectionTextWrapClass}>
              <p className={sectionTitleClass}>Export Format</p>
              <p className={sectionDescClass}>Default format for exported reports</p>
            </div>
            <select
              value={appSettings.exportFormat}
              onChange={(e) => handleAppSettingChange('exportFormat', e.target.value)}
              className={selectClass}
            >
              <option>PDF</option>
              <option>CSV</option>
              <option>JSON</option>
            </select>
          </div>

          <Separator className="bg-slate-200 dark:bg-slate-700" />

          <div className={sectionRowClass}>
            <div className={sectionTextWrapClass}>
              <p className={sectionTitleClass}>Data Retention</p>
              <p className={sectionDescClass}>How long to keep analysis data</p>
            </div>
            <select
              value={appSettings.dataRetention}
              onChange={(e) => handleAppSettingChange('dataRetention', e.target.value)}
              className={selectClass}
            >
              <option>5 minutes</option>
              <option>3 months</option>
              <option>6 months</option>
              <option>1 year</option>
              <option>Forever</option>
            </select>
          </div>

          <div className="flex justify-end">
            <Button
              onClick={handleSavePreferences}
              disabled={isSavingPreferences}
              className="h-11 rounded-lg bg-teal-500 px-5 text-white hover:bg-teal-600 dark:bg-teal-500 dark:hover:bg-teal-600"
            >
              <Save className="mr-2 h-4 w-4" />
              {isSavingPreferences ? 'Saving...' : 'Save App Settings'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Data Management */}
      <Card className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-800">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg font-semibold text-slate-900 dark:text-white">
            <FileDown className="h-5 w-5 text-emerald-500" />
            Data Management
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-5">
          <div className={sectionRowClass}>
            <div className={sectionTextWrapClass}>
              <p className={sectionTitleClass}>Export All Data</p>
              <p className={sectionDescClass}>
                Download all your patient records and analysis data
              </p>
            </div>
            <Button
              variant="outline"
              className="h-11 rounded-lg border-slate-300 px-4 dark:border-slate-600 dark:bg-transparent dark:text-white"
              disabled
            >
              <FileDown className="mr-2 h-4 w-4" />
              Export
            </Button>
          </div>

          <div className="rounded-xl border border-red-500/50 bg-red-950/10 p-4 dark:bg-red-900/10">
            <div className="mb-2 flex items-center gap-2">
              <ShieldAlert className="h-5 w-5 text-red-400" />
              <p className="font-semibold text-red-300">Danger Zone</p>
            </div>
            <p className="mb-4 text-sm text-red-300/90">
              Deleting your account will permanently remove all data and cannot be undone.
            </p>
            <Button
              variant="destructive"
              size="sm"
              className="rounded-lg"
              onClick={handleDeleteAccount}
            >
              Delete Account
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}