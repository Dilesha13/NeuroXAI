import { useEffect, useMemo, useState } from 'react';
import { Activity, Users, FileCheck, AlertCircle, Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { API_BASE_URL } from '../../config';

type DashboardSummary = {
  stats: {
    total_eeg_analyses: number;
    seizure_detections: number;
    normal_recordings: number;
    active_patients: number;
  };
  seizure_trends: Array<{ month: string; detections: number; normal: number }>;
  confidence_distribution: Array<{ range: string; count: number }>;
  distribution: Array<{ name: string; value: number }>;
  recent_analyses: Array<{
    record_id: number;
    inference_id: number;
    patient_id: number;
    patient_code: string;
    patient_name?: string | null;
    date?: string | null;
    duration: string;
    status: string;
    result: string;
    confidence: number;
    report_id?: number | null;
    download_url?: string | null;
  }>;
};

const PIE_COLORS: Record<string, string> = {
  Seizure: '#EF4444',
  Normal: '#10B981',
};

function formatDate(value?: string | null): string {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function getStatusBadge(status: string): string {
  if (status === 'Seizure Detected') {
    return 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200';
  }
  if (status === 'Review Needed' || status === 'Possible Seizure Activity — Review Needed') {
    return 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200';
  }
  return 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200';
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function loadDashboard() {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`${API_BASE_URL}/dashboard/summary`);
        if (!response.ok) {
          throw new Error('Failed to load dashboard data');
        }

        const payload: DashboardSummary = await response.json();

        if (isMounted) {
          setData(payload);
        }
      } catch (err) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Something went wrong');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    loadDashboard();

    return () => {
      isMounted = false;
    };
  }, []);

  const stats = useMemo(() => {
    const summary = data?.stats;

    return [
      {
        title: 'Total EEG Analyses',
        value: summary?.total_eeg_analyses ?? 0,
        icon: Activity,
        color: 'text-blue-600 dark:text-blue-400',
        bgColor: 'bg-blue-100 dark:bg-blue-900',
      },
      {
        title: 'Seizure Detections',
        value: summary?.seizure_detections ?? 0,
        icon: AlertCircle,
        color: 'text-red-600 dark:text-red-400',
        bgColor: 'bg-red-100 dark:bg-red-900',
      },
      {
        title: 'Normal Recordings',
        value: summary?.normal_recordings ?? 0,
        icon: FileCheck,
        color: 'text-green-600 dark:text-green-400',
        bgColor: 'bg-green-100 dark:bg-green-900',
      },
      {
        title: 'Active Patients',
        value: summary?.active_patients ?? 0,
        icon: Users,
        color: 'text-purple-600 dark:text-purple-400',
        bgColor: 'bg-purple-100 dark:bg-purple-900',
      },
    ];
  }, [data]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Overview of neonatal seizure detection analytics
        </p>
      </div>

      {loading ? (
        <Card className="dark:bg-slate-800 dark:border-gray-700">
          <CardContent className="p-12 flex items-center justify-center gap-3 text-gray-600 dark:text-gray-300">
            <Loader2 className="h-5 w-5 animate-spin" />
            Loading dashboard data...
          </CardContent>
        </Card>
      ) : error ? (
        <Card className="dark:bg-slate-800 dark:border-gray-700">
          <CardContent className="p-6 text-red-600 dark:text-red-300">{error}</CardContent>
        </Card>
      ) : data ? (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {stats.map((stat) => (
              <Card key={stat.title} className="dark:bg-slate-800 dark:border-gray-700">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                      <stat.icon className={`h-6 w-6 ${stat.color}`} />
                    </div>
                  </div>
                  <div className="mt-4">
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {stat.value.toLocaleString()}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{stat.title}</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="dark:bg-slate-800 dark:border-gray-700">
              <CardHeader>
                <CardTitle className="text-gray-900 dark:text-white">Seizure Detection Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={data.seizure_trends}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                    <XAxis dataKey="month" className="text-gray-600 dark:text-gray-400" />
                    <YAxis className="text-gray-600 dark:text-gray-400" allowDecimals={false} />
                    <Tooltip />
                    <Area
                      type="monotone"
                      dataKey="detections"
                      stackId="1"
                      stroke="#EF4444"
                      fill="#EF4444"
                      fillOpacity={0.6}
                      name="Seizure Detections"
                    />
                    <Area
                      type="monotone"
                      dataKey="normal"
                      stackId="1"
                      stroke="#10B981"
                      fill="#10B981"
                      fillOpacity={0.6}
                      name="Normal"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="dark:bg-slate-800 dark:border-gray-700">
              <CardHeader>
                <CardTitle className="text-gray-900 dark:text-white">Model Confidence Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={data.confidence_distribution}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                    <XAxis dataKey="range" className="text-gray-600 dark:text-gray-400" />
                    <YAxis className="text-gray-600 dark:text-gray-400" allowDecimals={false} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3B82F6" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="dark:bg-slate-800 dark:border-gray-700">
              <CardHeader>
                <CardTitle className="text-gray-900 dark:text-white">Analysis Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={data.distribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      dataKey="value"
                    >
                      {data.distribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={PIE_COLORS[entry.name] ?? '#94A3B8'} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>

                <div className="mt-4 space-y-2">
                  {data.distribution.map((entry) => (
                    <div className="flex items-center justify-between" key={entry.name}>
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: PIE_COLORS[entry.name] ?? '#94A3B8' }}
                        />
                        <span className="text-sm text-gray-600 dark:text-gray-400">{entry.name}</span>
                      </div>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {entry.value}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="dark:bg-slate-800 dark:border-gray-700 lg:col-span-2">
              <CardHeader>
                <CardTitle className="text-gray-900 dark:text-white">Recent Analyses</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {data.recent_analyses.map((analysis) => (
                    <div
                      key={analysis.inference_id}
                      className="flex items-center justify-between gap-4 p-4 rounded-lg border border-gray-200 dark:border-gray-700"
                    >
                      <div className="min-w-0">
                        <p className="font-semibold text-gray-900 dark:text-white">
                          {analysis.patient_code}
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400 truncate">
                          {analysis.patient_name ?? 'Unnamed patient'} • {formatDate(analysis.date)} •{' '}
                          {analysis.duration}
                        </p>
                      </div>

                      <div className="flex items-center gap-3">
                        <div className="text-right">
                          <p className="text-sm font-medium text-gray-900 dark:text-white">
                            {analysis.confidence.toFixed(1)}%
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">confidence</p>
                        </div>
                        <Badge className={getStatusBadge(analysis.status)}>{analysis.status}</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </>
      ) : null}
    </div>
  );
}