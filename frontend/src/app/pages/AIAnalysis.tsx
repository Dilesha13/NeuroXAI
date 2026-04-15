import { useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  CheckCircle,
  Activity,
  Clock,
  TrendingUp,
  Brain,
  FileText,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Progress } from '../components/ui/progress';
import { Button } from '../components/ui/button';
import { useNavigate } from 'react-router-dom';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine,
} from 'recharts';

type TimelineItem = {
  window_index: number;
  start_sec: number;
  end_sec: number;
  probability: number;
  predicted_label: number;
};

type StoredResult = {
  inference_id?: number;
  patient?: {
    inputPatientCode?: string;
    backendPatientId?: number;
    recordingDate?: string;
  };
  upload?: {
    id?: number;
    patient_id?: number;
    original_filename?: string;
    stored_path?: string;
    status?: string;
    duration_seconds?: number | null;
    sampling_rate_original?: number | null;
    created_at?: string;
  };
  analysis?: {
    seizure_detected?: boolean;
    overall_prediction?: string;
    overall_prediction_code?: string;
    probability_score?: number;
    confidence_level?: string;
    duration_minutes?: number;
    num_windows?: number;
    num_seizure_windows?: number;
    mean_probability?: number;
    threshold_used?: number;
    overall_summary?: string;
    estimated_seizure_duration_minutes?: number;
  };
  timeline?: TimelineItem[];
  explainability?: {
    top_channels?: Array<{ channel?: string; score?: number; importance?: number }> | string[];
    temporal_attention?: Array<{ window_index: number; attention: number }>;
    gat_edges?: Array<{ source: string; target: string; weight: number }>;
    seizure_ranges?: Array<{
      start_sec?: number;
      end_sec?: number;
      start_min?: number;
      end_min?: number;
    }>;
    probability_timeline?: number[];
    saliency_path?: string;
    temporal_attention_path?: string;
    gat_attention_path?: string;
    summary_text?: string;
    saliency_available?: boolean;
  };
  model?: {
    name?: string;
    threshold?: number;
    checkpoint?: string;
  };
};

export default function AIAnalysis() {
  const navigate = useNavigate();
  const [storedResult, setStoredResult] = useState<StoredResult | null>(null);

  useEffect(() => {
    const raw = localStorage.getItem('latestInferenceResult');
    if (raw) {
      try {
        setStoredResult(JSON.parse(raw));
      } catch (error) {
        console.error('Failed to parse latestInferenceResult:', error);
      }
    }
  }, []);

  const patient = storedResult?.patient;
  const upload = storedResult?.upload;
  const analysis = storedResult?.analysis;
  const timeline = storedResult?.timeline ?? [];
  const explainability = storedResult?.explainability;
  const model = storedResult?.model;
  const inferenceId = storedResult?.inference_id;

  const thresholdValue = (analysis?.threshold_used ?? model?.threshold ?? 0.21) as number;

  const timelineData = useMemo(() => {
    return timeline.map((item) => ({
      time: Number((item.start_sec / 60).toFixed(2)),
      startSec: item.start_sec,
      endSec: item.end_sec,
      probability: item.probability,
      predictedLabel: item.predicted_label,
      aboveThreshold: item.probability >= thresholdValue,
    }));
  }, [timeline, thresholdValue]);

  const seizureWindows = useMemo(() => {
    return timeline.filter((item) => item.predicted_label === 1);
  }, [timeline]);

  const seizureDetected =
    analysis?.seizure_detected ??
    analysis?.overall_prediction?.toLowerCase().includes('seizure detected') ??
    false;

  const confidenceLevel = (analysis?.confidence_level ?? '').toLowerCase();
  const isLowConfidencePositive =
    seizureDetected && confidenceLevel.includes('low') && !confidenceLevel.includes('very high');

  const displayStatus = !seizureDetected
    ? 'No Seizure Detected'
    : isLowConfidencePositive
    ? 'Possible Seizure Activity — Review Needed'
    : 'Seizure Detected';

  const statusTone: 'negative' | 'warning' | 'positive' = !seizureDetected
    ? 'negative'
    : isLowConfidencePositive
    ? 'warning'
    : 'positive';

  const maxProbabilityPercent = (((analysis?.probability_score ?? 0) as number) * 100).toFixed(1);
  const meanProbabilityPercent = (((analysis?.mean_probability ?? 0) as number) * 100).toFixed(1);
  const thresholdPercent = (thresholdValue * 100).toFixed(0);

  const totalWindows = analysis?.num_windows ?? timelineData.length;
  const seizureWindowCount = analysis?.num_seizure_windows ?? seizureWindows.length;

  const recordingDurationMinutes = useMemo(() => {
    if (analysis?.duration_minutes && analysis.duration_minutes > 0) {
      return analysis.duration_minutes.toFixed(1);
    }

    if (upload?.duration_seconds && upload.duration_seconds > 0) {
      return (upload.duration_seconds / 60).toFixed(1);
    }

    if (timelineData.length > 0) {
      const last = timelineData[timelineData.length - 1];
      return (last.endSec / 60).toFixed(1);
    }

    return 'N/A';
  }, [analysis, upload, timelineData]);

  const peakWindow = useMemo(() => {
    if (!timelineData.length) return null;
    return timelineData.reduce((best, current) =>
      current.probability > best.probability ? current : best
    );
  }, [timelineData]);

  const firstSeizureWindow = seizureWindows.length > 0 ? seizureWindows[0] : null;
  const lastSeizureWindow =
    seizureWindows.length > 0 ? seizureWindows[seizureWindows.length - 1] : null;

  const affectedDurationMinutes = useMemo(() => {
    if (typeof analysis?.estimated_seizure_duration_minutes === 'number') {
      return analysis.estimated_seizure_duration_minutes.toFixed(1);
    }

    if (firstSeizureWindow && lastSeizureWindow) {
      return ((lastSeizureWindow.end_sec - firstSeizureWindow.start_sec) / 60).toFixed(1);
    }

    return '0.0';
  }, [analysis, firstSeizureWindow, lastSeizureWindow]);

  const analysisTimeText = patient?.recordingDate
    ? new Date(patient.recordingDate).toLocaleString()
    : upload?.created_at
    ? new Date(upload.created_at).toLocaleString()
    : 'N/A';

  const topChannelText = useMemo(() => {
    const topChannels = explainability?.top_channels ?? [];
    if (!Array.isArray(topChannels) || topChannels.length === 0) return 'N/A';

    const first = topChannels[0];
    if (typeof first === 'string') return first;
    return first.channel ?? 'N/A';
  }, [explainability]);

  const strongestTemporalSegment = useMemo(() => {
    const segments = explainability?.temporal_attention ?? [];
    if (!Array.isArray(segments) || segments.length === 0) return null;

    return segments.reduce((best, current) =>
      current.attention > best.attention ? current : best
    );
  }, [explainability]);

  const strongestEdge = useMemo(() => {
    const edges = explainability?.gat_edges ?? [];
    if (!Array.isArray(edges) || edges.length === 0) return null;

    return edges.reduce((best, current) =>
      current.weight > best.weight ? current : best
    );
  }, [explainability]);

  const handleViewExplainability = () => {
    navigate('/app/explainability');
  };

  const handleViewReport = () => {
    if (!inferenceId) {
      alert('Inference ID not found. Please re-run the analysis.');
      return;
    }
    navigate(`/app/reports?inference_id=${inferenceId}`);
  };

  if (!storedResult || !analysis) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">AI Analysis Results</h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            No completed backend analysis result was found.
          </p>
        </div>

        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardContent className="p-6">
            <p className="text-gray-700 dark:text-gray-300">
              Please upload an EEG recording first, then return to this page after processing.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">AI Analysis Results</h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            Backend-generated inference output for Patient {patient?.inputPatientCode ?? 'N/A'}
          </p>
        </div>


        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.35fr_0.9fr]">
          <Card className="overflow-hidden border border-orange-300/70 bg-[linear-gradient(135deg,#EAF6FB_0%,#D9EDF7_50%,#C9E3F0_100%)] shadow-2xl dark:border-orange-400/30 dark:bg-[linear-gradient(135deg,#4A8A9E_0%,#3A5A78_55%,#2A4A5E_100%)]">
            <CardContent className="p-0">
              <div className="relative overflow-hidden px-7 py-7">
                <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-white/40 blur-2xl dark:bg-white/10" />
                <div className="absolute -bottom-10 -left-10 h-40 w-40 rounded-full bg-sky-200/40 blur-2xl dark:bg-black/10" />

                <div className="relative space-y-6">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div className="space-y-2">
                      <div className="inline-flex items-center gap-2 rounded-full border border-orange-300/70 bg-white/65 px-3 py-1 text-sm font-medium text-sky-900 dark:border-orange-400/30 dark:bg-[rgba(42,74,94,0.35)] dark:text-white/90">
                        <Activity className="h-4 w-4" />
                        Quick Clinical Interpretation
                      </div>

                      <h2 className="max-w-3xl text-3xl font-bold leading-tight text-slate-900 md:text-4xl dark:text-white">
                        {seizureDetected
                          ? isLowConfidencePositive
                            ? 'Possible seizure activity was flagged by the AI'
                            : 'Strong seizure-related activity was detected by the AI'
                          : 'No seizure activity was detected by the AI'}
                      </h2>

                      <p className="max-w-2xl text-sm leading-6 text-slate-700 md:text-base dark:text-white/85">
                        {seizureDetected
                          ? isLowConfidencePositive
                            ? 'The model detected seizure-like patterns, but confidence is limited. This result should be reviewed carefully together with EEG waveform inspection and clinical context.'
                            : 'The model identified strong seizure-related activity. The key indicators below show what most strongly supported the AI result.'
                          : 'The model did not find seizure activity above the configured threshold. The indicators below summarize how the AI reached this result.'}
                      </p>
                    </div>

                    <div className="rounded-2xl border border-orange-300/70 bg-white/60 px-4 py-3 text-center backdrop-blur-sm dark:border-orange-400/30 dark:bg-[rgba(42,74,94,0.35)]">
                      <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-white/75">
                        Model Status
                      </p>
                      <p className="mt-1 text-lg font-bold text-slate-900 dark:text-white">
                        {displayStatus}
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                    <div className="rounded-2xl border border-orange-300/70 bg-white/60 p-5 backdrop-blur-sm dark:border-orange-400/20 dark:bg-[rgba(42,74,94,0.42)]">
                      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-white/60">
                        Peak Probability
                      </p>
                      <p className="mt-2 text-2xl font-bold text-slate-900 dark:text-white">
                        {maxProbabilityPercent}%
                      </p>
                      <p className="mt-1 text-sm text-slate-600 dark:text-white/70">
                        Highest seizure likelihood seen
                      </p>
                    </div>

                    <div className="rounded-2xl border border-orange-300/70 bg-white/60 p-5 backdrop-blur-sm dark:border-orange-400/20 dark:bg-[rgba(42,74,94,0.42)]">
                      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-white/60">
                        Decision Threshold
                      </p>
                      <p className="mt-2 text-2xl font-bold text-slate-900 dark:text-white">
                        {thresholdPercent}%
                      </p>
                      <p className="mt-1 text-sm text-slate-600 dark:text-white/70">
                        Cutoff used to flag activity
                      </p>
                    </div>

                    <div className="rounded-2xl border border-orange-300/70 bg-white/60 p-5 backdrop-blur-sm dark:border-orange-400/20 dark:bg-[rgba(42,74,94,0.42)]">
                      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-white/60">
                        Positive Windows
                      </p>
                      <p className="mt-2 text-2xl font-bold text-slate-900 dark:text-white">
                        {seizureWindowCount} / {totalWindows}
                      </p>
                      <p className="mt-1 text-sm text-slate-600 dark:text-white/70">
                        Analysed segments above threshold
                      </p>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-orange-300/70 bg-white/60 px-5 py-4 text-sm text-slate-700 backdrop-blur-sm dark:border-orange-400/20 dark:bg-[rgba(42,74,94,0.50)] dark:text-white/90">
                    <span className="font-semibold text-slate-900 dark:text-white">Simple meaning:</span>{' '}
                    This result summarizes how strongly the model believed seizure-related activity was present in the EEG.
                    It is intended for decision support and should be interpreted together with waveform review and clinical judgment.
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border border-orange-300/70 bg-gradient-to-b from-slate-50 to-orange-50 shadow-xl dark:border-orange-400/30 dark:from-slate-900 dark:to-orange-950/50">
            <CardContent className="p-6">
              <div className="space-y-5">
                <div className="space-y-2">
                  <div className="inline-flex items-center gap-2 rounded-full border border-orange-300 bg-orange-100 px-3 py-1 text-sm font-medium text-orange-800 dark:border-orange-400/20 dark:bg-orange-500/10 dark:text-orange-200">
                    <AlertCircle className="h-4 w-4" />
                    How to Read This Result
                  </div>

                  <p className="text-sm leading-6 text-slate-700 dark:text-slate-300">
                    Use this page as a quick guide to understand <span className="font-semibold text-slate-900 dark:text-white">what</span> the AI
                    concluded and <span className="font-semibold text-slate-900 dark:text-white">how strongly</span> it supported that conclusion.
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="rounded-xl border border-slate-200 bg-white/80 p-4 dark:border-white/10 dark:bg-white/5">
                    <p className="text-sm font-semibold text-slate-900 dark:text-white">Status</p>
                    <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                      Shows the overall AI decision for this EEG recording.
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-white/80 p-4 dark:border-white/10 dark:bg-white/5">
                    <p className="text-sm font-semibold text-slate-900 dark:text-white">Peak Probability</p>
                    <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                      Shows the strongest seizure likelihood seen in any analysed segment.
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-white/80 p-4 dark:border-white/10 dark:bg-white/5">
                    <p className="text-sm font-semibold text-slate-900 dark:text-white">Threshold</p>
                    <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                      Shows the cutoff used by the system to decide whether activity should be flagged.
                    </p>
                  </div>
                </div>

                <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 dark:border-amber-400/30 dark:bg-amber-500/10">
                  <p className="text-sm font-semibold text-amber-800 dark:text-amber-200">
                    Important
                  </p>
                  <p className="mt-1 text-sm leading-6 text-amber-900/80 dark:text-amber-100/90">
                    This AI result supports review only. It should assist clinical interpretation, not replace expert judgment.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
        <div className="flex flex-wrap gap-3">
          <Button
            onClick={handleViewExplainability}
            className="bg-blue-600 text-white hover:bg-blue-700 dark:bg-teal-600 dark:hover:bg-teal-700"
          >
            <Brain className="mr-2 h-4 w-4" />
            View Explainability
          </Button>

          <Button
            onClick={handleViewReport}
            variant="outline"
            className="dark:border-gray-700"
          >
            <FileText className="mr-2 h-4 w-4" />
            View Report
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-blue-100 p-3 dark:bg-blue-900">
                <Activity className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Patient ID</p>
                <p className="text-xl font-bold text-gray-900 dark:text-white">
                  {patient?.inputPatientCode ?? 'N/A'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-purple-100 p-3 dark:bg-purple-900">
                <Clock className="h-6 w-6 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Duration</p>
                <p className="text-xl font-bold text-gray-900 dark:text-white">
                  {recordingDurationMinutes} min
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-teal-100 p-3 dark:bg-teal-900">
                <TrendingUp className="h-6 w-6 text-teal-600 dark:text-teal-400" />
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Analysis Time</p>
                <p className="text-xl font-bold text-gray-900 dark:text-white">{analysisTimeText}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card
        className={`border-2 dark:border-gray-700 dark:bg-slate-800 ${
          statusTone === 'negative'
            ? 'border-green-200 dark:border-green-900'
            : statusTone === 'warning'
            ? 'border-yellow-200 dark:border-yellow-900'
            : 'border-red-200 dark:border-red-900'
        }`}
      >
        <CardHeader>
          <CardTitle className="flex items-center gap-3 text-gray-900 dark:text-white">
            {statusTone === 'negative' ? (
              <CheckCircle className="h-6 w-6 text-green-600 dark:text-green-400" />
            ) : statusTone === 'warning' ? (
              <AlertCircle className="h-6 w-6 text-yellow-600 dark:text-yellow-400" />
            ) : (
              <AlertCircle className="h-6 w-6 text-red-600 dark:text-red-400" />
            )}
            Prediction Result
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-4">
            <div>
              <p className="mb-2 text-sm text-gray-600 dark:text-gray-400">Status</p>
              <Badge
                className={
                  statusTone === 'negative'
                    ? 'bg-green-100 px-4 py-2 text-lg text-green-800 dark:bg-green-900 dark:text-green-200'
                    : statusTone === 'warning'
                    ? 'bg-yellow-100 px-4 py-2 text-lg text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                    : 'bg-red-100 px-4 py-2 text-lg text-red-800 dark:bg-red-900 dark:text-red-200'
                }
              >
                {displayStatus}
              </Badge>
            </div>

            <div>
              <p className="mb-2 text-sm text-gray-600 dark:text-gray-400">Peak Probability</p>
              <p
                className={`text-3xl font-bold ${
                  statusTone === 'negative'
                    ? 'text-green-600 dark:text-green-400'
                    : statusTone === 'warning'
                    ? 'text-yellow-600 dark:text-yellow-400'
                    : 'text-red-600 dark:text-red-400'
                }`}
              >
                {maxProbabilityPercent}%
              </p>
            </div>

            <div>
              <p className="mb-2 text-sm text-gray-600 dark:text-gray-400">Mean Probability</p>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">
                {meanProbabilityPercent}%
              </p>
            </div>

            <div>
              <p className="mb-2 text-sm text-gray-600 dark:text-gray-400">Confidence Level</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {analysis?.confidence_level ?? 'N/A'}
              </p>
            </div>
          </div>

          <div>
            <p className="mb-2 text-sm text-gray-600 dark:text-gray-400">Decision Threshold</p>
            <div className="space-y-2">
              <Progress value={Number(maxProbabilityPercent)} className="h-3" />
              <p className="text-sm font-semibold text-gray-900 dark:text-white">
                Threshold used: {thresholdPercent}%
              </p>
            </div>
          </div>

          <div
            className={`rounded-lg border p-4 ${
              statusTone === 'negative'
                ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
                : statusTone === 'warning'
                ? 'border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20'
                : 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
            }`}
          >
            <p
              className={`text-sm ${
                statusTone === 'negative'
                  ? 'text-green-800 dark:text-green-200'
                  : statusTone === 'warning'
                  ? 'text-yellow-800 dark:text-yellow-200'
                  : 'text-red-800 dark:text-red-200'
              }`}
            >
              <strong>Research Prototype Note:</strong>{' '}
              {analysis?.overall_summary ??
                (!seizureDetected
                  ? 'The model did not identify windows exceeding the configured seizure threshold in this EEG recording. This output is intended for decision-support demonstration and should be interpreted alongside expert review.'
                  : isLowConfidencePositive
                  ? 'The model detected seizure-like activity above the deployment threshold, but confidence is low. Clinical review is recommended before interpretation.'
                  : 'The model identified strong seizure-related activity in this EEG recording. This output is intended for decision-support demonstration and should be reviewed together with clinical expertise.')}
            </p>
          </div>
        </CardContent>
      </Card>

      <Card className="dark:border-gray-700 dark:bg-slate-800">
        <CardHeader>
          <CardTitle className="text-gray-900 dark:text-white">Seizure Probability Timeline</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
              <XAxis
                dataKey="time"
                label={{ value: 'Time (minutes)', position: 'insideBottom', offset: -5 }}
                className="text-gray-600 dark:text-gray-400"
              />
              <YAxis
                domain={[0, 1]}
                label={{ value: 'Probability', angle: -90, position: 'insideLeft' }}
                className="text-gray-600 dark:text-gray-400"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                labelFormatter={(label) => `Time: ${label} min`}
              />
              <ReferenceLine y={thresholdValue} stroke="#F59E0B" strokeDasharray="5 5" />
              <Area
                type="monotone"
                dataKey="probability"
                stroke="#EF4444"
                fill="#EF4444"
                fillOpacity={0.3}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>

          <div className="mt-4 rounded-lg border border-yellow-200 bg-yellow-50 p-3 dark:border-yellow-800 dark:bg-yellow-900/20">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              {peakWindow
                ? `Peak model probability occurred around minute ${peakWindow.time} with a probability of ${(
                    peakWindow.probability * 100
                  ).toFixed(1)}%. The dashed line represents the decision threshold.`
                : 'No timeline data available for this recording.'}
            </p>
          </div>
        </CardContent>
      </Card>

      <Card className="dark:border-gray-700 dark:bg-slate-800">
        <CardHeader>
          <CardTitle className="text-gray-900 dark:text-white">Window-Level Summary</CardTitle>
        </CardHeader>
        <CardContent>
          {timelineData.length === 0 ? (
            <p className="text-gray-600 dark:text-gray-400">No window-level inference data available.</p>
          ) : (
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                <XAxis
                  dataKey="time"
                  label={{ value: 'Time (minutes)', position: 'insideBottom', offset: -5 }}
                  className="text-gray-600 dark:text-gray-400"
                />
                <YAxis
                  domain={[0, 1]}
                  label={{ value: 'Probability', angle: -90, position: 'insideLeft' }}
                  className="text-gray-600 dark:text-gray-400"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                  }}
                  formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                />
                <ReferenceLine y={thresholdValue} stroke="#F59E0B" strokeDasharray="5 5" />
                <Line
                  type="monotone"
                  dataKey="probability"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={false}
                  name="Window Probability"
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardContent className="p-6">
            <p className="mb-1 text-sm text-gray-600 dark:text-gray-400">Peak Time</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {peakWindow ? `${peakWindow.time} min` : 'N/A'}
            </p>
          </CardContent>
        </Card>

        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardContent className="p-6">
            <p className="mb-1 text-sm text-gray-600 dark:text-gray-400">Estimated Event Span</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {affectedDurationMinutes} min
            </p>
          </CardContent>
        </Card>

        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardContent className="p-6">
            <p className="mb-1 text-sm text-gray-600 dark:text-gray-400">Positive Windows</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {seizureWindowCount} / {totalWindows}
            </p>
          </CardContent>
        </Card>

        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardContent className="p-6">
            <p className="mb-1 text-sm text-gray-600 dark:text-gray-400">Model</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {model?.name ?? 'MST-GAT'}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardContent className="p-6">
            <p className="mb-1 text-sm text-gray-600 dark:text-gray-400">Top Channel</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">{topChannelText}</p>
          </CardContent>
        </Card>

        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardContent className="p-6">
            <p className="mb-1 text-sm text-gray-600 dark:text-gray-400">Strongest Temporal Segment</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">
              {strongestTemporalSegment
                ? `Segment ${strongestTemporalSegment.window_index + 1}`
                : 'N/A'}
            </p>
          </CardContent>
        </Card>

        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardContent className="p-6">
            <p className="mb-1 text-sm text-gray-600 dark:text-gray-400">Strongest Graph Edge</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">
              {strongestEdge ? `${strongestEdge.source} → ${strongestEdge.target}` : 'N/A'}
            </p>
          </CardContent>
        </Card>
      </div>

      {explainability?.summary_text && (
        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardHeader>
            <CardTitle className="text-gray-900 dark:text-white">Explainability Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-700 dark:text-gray-300">{explainability.summary_text}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}