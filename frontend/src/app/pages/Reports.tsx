import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Download,
  Printer,
  Share2,
  Calendar,
  User,
  Activity,
  Loader2,
  AlertCircle,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';
const BACKEND_BASE_URL = 'http://127.0.0.1:8000';

type TopChannel = {
  channel: string;
  score: number;
  region?: string;
};

type SeizureRange = {
  start_sec?: number;
  end_sec?: number;
  start_min?: number;
  end_min?: number;
};

type ReportData = {
  patient_id: string;
  eeg_file: string;
  recording_date: string;
  duration_minutes?: number | null;
  prediction: string;
  confidence_score: number;
  confidence_label: string;
  threshold: number;
  mean_probability: number;
  num_windows: number;
  num_seizure_windows: number;
  estimated_seizure_duration_minutes: number;
  probability_timeline: number[];
  seizure_ranges: SeizureRange[];
  top_channels: TopChannel[];
  dominant_region?: string | null;
  clinical_findings: string;
  recommendations: string[];
  explanation_summary: string;
  disclaimer: string;
};

type ReportPreviewResponse = {
  inference_id: number;
  report_data: ReportData;
};

type GenerateReportResponse = {
  report_id: number;
  report_path: string;
  download_url: string;
  report_data: ReportData;
};

function formatPercent(score: number): string {
  return `${(score * 100).toFixed(1)}%`;
}

function formatPredictionVariant(
  prediction: string
): 'default' | 'secondary' | 'destructive' | 'outline' {
  return prediction.toLowerCase().includes('seizure') &&
    prediction.toLowerCase().includes('detected')
    ? 'destructive'
    : 'secondary';
}

function buildProbabilityChartData(report: ReportData) {
  const probs = report.probability_timeline || [];
  if (!probs.length) return [];

  const duration = typeof report.duration_minutes === 'number' ? report.duration_minutes : null;
  const total = probs.length;

  return probs.map((value, index) => {
    const time =
      duration && total > 1
        ? Number(((index / (total - 1)) * duration).toFixed(1))
        : index + 1;

    return {
      time,
      prob: Number((value * 100).toFixed(1)),
    };
  });
}

function buildChannelChartData(report: ReportData) {
  return (report.top_channels || []).slice(0, 6).map((item) => ({
    channel: item.channel,
    importance: Number((item.score * 100).toFixed(1)),
  }));
}

export default function Reports() {
  const [searchParams] = useSearchParams();
  const inferenceId = searchParams.get('inference_id');

  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [reportId, setReportId] = useState<number | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const probabilityChartData = useMemo(
    () => (reportData ? buildProbabilityChartData(reportData) : []),
    [reportData]
  );

  const channelChartData = useMemo(
    () => (reportData ? buildChannelChartData(reportData) : []),
    [reportData]
  );

  useEffect(() => {
    const fetchReportPreview = async () => {
      if (!inferenceId) {
        setError('Missing inference_id in URL.');
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`${API_BASE_URL}/reports/${inferenceId}`);
        if (!response.ok) {
          throw new Error(`Failed to load report preview (${response.status})`);
        }

        const data: ReportPreviewResponse = await response.json();
        setReportData(data.report_data);
      } catch (err) {
        console.error(err);
        setError('Failed to load report preview.');
      } finally {
        setLoading(false);
      }
    };

    fetchReportPreview();
  }, [inferenceId]);

  const handleExportPdf = async () => {
    if (!inferenceId) return;

    try {
      setExporting(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/reports/${inferenceId}/generate`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Failed to generate PDF (${response.status})`);
      }

      const data: GenerateReportResponse = await response.json();

      const fullDownloadUrl = `${BACKEND_BASE_URL}${data.download_url}`;

      setReportData(data.report_data);
      setReportId(data.report_id);
      setDownloadUrl(fullDownloadUrl);

      window.open(fullDownloadUrl, '_blank');
    } catch (err) {
      console.error(err);
      setError('Failed to generate PDF report.');
    } finally {
      setExporting(false);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  const handleShare = async () => {
    if (!reportData) return;

    const shareText = `NeuroXAI report for patient ${reportData.patient_id} - ${reportData.prediction}`;
    const shareUrl = window.location.href;

    try {
      if (navigator.share) {
        await navigator.share({
          title: 'NeuroXAI Analysis Report',
          text: shareText,
          url: shareUrl,
        });
      } else {
        await navigator.clipboard.writeText(shareUrl);
        alert('Report link copied to clipboard.');
      }
    } catch (err) {
      console.error(err);
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <div className="flex items-center gap-3 text-gray-700 dark:text-gray-300">
          <Loader2 className="h-5 w-5 animate-spin" />
          <span>Loading report...</span>
        </div>
      </div>
    );
  }

  if (error || !reportData) {
    return (
      <div className="space-y-6">
        <Card className="border-red-200 dark:border-red-800 dark:bg-slate-800">
          <CardContent className="p-8">
            <div className="flex items-center gap-3 text-red-700 dark:text-red-300">
              <AlertCircle className="h-5 w-5" />
              <div>
                <p className="font-semibold">Unable to load report</p>
                <p className="mt-1 text-sm">{error || 'No report data available.'}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const seizureDetected =
    reportData.prediction.toLowerCase().includes('seizure') &&
    reportData.prediction.toLowerCase().includes('detected');

  const highlightedRange =
    reportData.seizure_ranges && reportData.seizure_ranges.length > 0
      ? reportData.seizure_ranges[0]
      : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Analysis Report</h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            Comprehensive report for Patient {reportData.patient_id}
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          <Button variant="outline" className="dark:border-gray-700" onClick={handlePrint}>
            <Printer className="mr-2 h-4 w-4" />
            Print
          </Button>

          <Button variant="outline" className="dark:border-gray-700" onClick={handleShare}>
            <Share2 className="mr-2 h-4 w-4" />
            Share
          </Button>

          <Button
            className="bg-blue-600 text-white hover:bg-blue-700 dark:bg-teal-600 dark:hover:bg-teal-700"
            onClick={handleExportPdf}
            disabled={exporting}
          >
            {exporting ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Download className="mr-2 h-4 w-4" />
            )}
            {exporting ? 'Generating PDF...' : 'Export PDF'}
          </Button>
        </div>
      </div>

      <Card className="dark:border-gray-700 dark:bg-slate-800">
        <CardContent className="p-8">
          <div className="mb-6 flex items-start justify-between gap-4">
            <div>
              <h2 className="mb-2 text-2xl font-bold text-gray-900 dark:text-white">
                Neonatal EEG Seizure Detection Report
              </h2>
              <p className="text-gray-600 dark:text-gray-400">Generated by NeuroXAI Platform</p>
            </div>

            <div className="text-right">
              <Badge
                variant={formatPredictionVariant(reportData.prediction)}
                className="px-4 py-2 text-lg"
              >
                {reportData.prediction}
              </Badge>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 border-t border-gray-200 pt-6 dark:border-gray-700 md:grid-cols-3">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-blue-100 p-3 dark:bg-blue-900">
                <User className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Patient ID</p>
                <p className="font-semibold text-gray-900 dark:text-white">{reportData.patient_id}</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-purple-100 p-3 dark:bg-purple-900">
                <Calendar className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Recording Date</p>
                <p className="font-semibold text-gray-900 dark:text-white">
                  {reportData.recording_date}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-teal-100 p-3 dark:bg-teal-900">
                <Activity className="h-5 w-5 text-teal-600 dark:text-teal-400" />
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Duration</p>
                <p className="font-semibold text-gray-900 dark:text-white">
                  {typeof reportData.duration_minutes === 'number'
                    ? `${reportData.duration_minutes.toFixed(1)} minutes`
                    : 'N/A'}
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="dark:border-gray-700 dark:bg-slate-800">
        <CardHeader>
          <CardTitle className="text-gray-900 dark:text-white">Detection Summary</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <div
              className={`rounded-lg border p-6 ${
                seizureDetected
                  ? 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
                  : 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
              }`}
            >
              <p
                className={`mb-1 text-sm ${
                  seizureDetected
                    ? 'text-red-600 dark:text-red-400'
                    : 'text-green-600 dark:text-green-400'
                }`}
              >
                Prediction
              </p>
              <p
                className={`text-2xl font-bold ${
                  seizureDetected
                    ? 'text-red-700 dark:text-red-300'
                    : 'text-green-700 dark:text-green-300'
                }`}
              >
                {reportData.prediction}
              </p>
            </div>

            <div className="rounded-lg border border-blue-200 bg-blue-50 p-6 dark:border-blue-800 dark:bg-blue-900/20">
              <p className="mb-1 text-sm text-blue-600 dark:text-blue-400">Confidence Score</p>
              <p className="text-2xl font-bold text-blue-700 dark:text-blue-300">
                {formatPercent(reportData.confidence_score)}
              </p>
              <p className="mt-1 text-xs text-blue-600 dark:text-blue-400">
                {reportData.confidence_label}
              </p>
            </div>

            <div className="rounded-lg border border-purple-200 bg-purple-50 p-6 dark:border-purple-800 dark:bg-purple-900/20">
              <p className="mb-1 text-sm text-purple-600 dark:text-purple-400">Seizure Duration</p>
              <p className="text-2xl font-bold text-purple-700 dark:text-purple-300">
                ~{reportData.estimated_seizure_duration_minutes.toFixed(1)} minutes
              </p>
            </div>
          </div>

          <div className="prose max-w-none dark:prose-invert">
            <h4 className="mb-3 font-semibold text-gray-900 dark:text-white">Clinical Findings:</h4>
            <div className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
              <p>{reportData.clinical_findings}</p>

              {highlightedRange && (
                <p>
                  The most clearly highlighted abnormal interval spans approximately{' '}
                  <span className="font-semibold">
                    {highlightedRange.start_min?.toFixed?.(1) ?? highlightedRange.start_min ?? 0} to{' '}
                    {highlightedRange.end_min?.toFixed?.(1) ?? highlightedRange.end_min ?? 0} minutes
                  </span>{' '}
                  of the recording.
                </p>
              )}

              {reportData.dominant_region && (
                <p>
                  Explainability signals suggest strongest contribution from the{' '}
                  <span className="font-semibold">{reportData.dominant_region}</span> region.
                </p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardHeader>
            <CardTitle className="text-gray-900 dark:text-white">Seizure Probability Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            {probabilityChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={probabilityChartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="time" label={{ value: 'Time (min)', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                    }}
                  />
                  <Line type="monotone" dataKey="prob" stroke="#EF4444" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[250px] items-center justify-center text-sm text-gray-500 dark:text-gray-400">
                No probability timeline available
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardHeader>
            <CardTitle className="text-gray-900 dark:text-white">Channel Contribution Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            {channelChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={channelChartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="channel" />
                  <YAxis label={{ value: 'Importance (%)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                    }}
                  />
                  <Bar dataKey="importance" fill="#3B82F6" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[250px] items-center justify-center text-sm text-gray-500 dark:text-gray-400">
                No channel contribution data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="dark:border-gray-700 dark:bg-slate-800">
        <CardHeader>
          <CardTitle className="text-gray-900 dark:text-white">Explainability Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 text-sm text-gray-700 dark:text-gray-300">
            <p>{reportData.explanation_summary}</p>

            {reportData.top_channels.length > 0 && (
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                {reportData.top_channels.slice(0, 6).map((item) => (
                  <div
                    key={item.channel}
                    className="rounded-lg border border-gray-200 p-3 dark:border-gray-700"
                  >
                    <p className="font-semibold text-gray-900 dark:text-white">{item.channel}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {item.region || 'Unknown region'}
                    </p>
                    <p className="mt-1 text-sm">{formatPercent(item.score)} importance</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="dark:border-gray-700 dark:bg-slate-800">
        <CardHeader>
          <CardTitle className="text-gray-900 dark:text-white">Clinical Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {reportData.recommendations.map((item, index) => {
              const styles = [
                'border-red-500 bg-red-50 dark:border-red-400 dark:bg-red-900/20 text-red-800 dark:text-red-300',
                'border-blue-500 bg-blue-50 dark:border-blue-400 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300',
                'border-purple-500 bg-purple-50 dark:border-purple-400 dark:bg-purple-900/20 text-purple-800 dark:text-purple-300',
                'border-amber-500 bg-amber-50 dark:border-amber-400 dark:bg-amber-900/20 text-amber-800 dark:text-amber-300',
              ];
              const style = styles[index % styles.length];

              return (
                <div key={`${item}-${index}`} className={`border-l-4 p-4 ${style}`}>
                  <p className="text-sm font-medium">{item}</p>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <Card className="dark:border-gray-700 dark:bg-slate-800">
        <CardContent className="p-6">
          <div className="space-y-2 text-center text-sm text-gray-600 dark:text-gray-400">
            <p>This report was generated by NeuroXAI AI-Powered Neonatal Seizure Detection Platform</p>
            <p>
              Report Generated: {new Date().toLocaleString()}
              {reportId ? ` | Report ID: ${reportId}` : ''}
            </p>
            {downloadUrl && (
              <p className="text-xs">
                PDF available at: <span className="font-medium">{downloadUrl}</span>
              </p>
            )}
            <p className="mt-4 border-t border-gray-200 pt-2 text-xs dark:border-gray-700">
              {reportData.disclaimer}
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}