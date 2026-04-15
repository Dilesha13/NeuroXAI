import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Brain,
  Info,
  TrendingUp,
  Zap,
  Network,
  FileText,
  Activity,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Progress } from '../components/ui/progress';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';

type TimelineItem = {
  window_index: number;
  start_sec: number;
  end_sec: number;
  probability: number;
  predicted_label: number;
};

type TemporalAttentionItem = {
  window_index: number;
  attention: number;
};

type TopChannelItem = {
  channel: string;
  importance?: number;
  score?: number;
};

type GatEdgeItem = {
  source: string;
  target: string;
  weight: number;
};

type ExplainabilityData = {
  top_channels?: TopChannelItem[] | string[];
  temporal_attention?: TemporalAttentionItem[];
  gat_edges?: GatEdgeItem[];
  summary_text?: string;
  saliency_available?: boolean;
  saliency_path?: string;
  temporal_attention_path?: string;
  gat_attention_path?: string;
};

type StoredResult = {
  inference_id?: number;
  patient?: {
    inputPatientCode?: string;
    backendPatientId?: number;
    recordingDate?: string;
  };
  analysis?: {
    seizure_detected?: boolean;
    overall_prediction?: string;
    probability_score?: number;
    confidence_level?: string;
    duration_minutes?: number;
    num_windows?: number;
    num_seizure_windows?: number;
    mean_probability?: number;
    threshold_used?: number;
    overall_summary?: string;
  };
  timeline?: TimelineItem[];
  explainability?: ExplainabilityData;
  model?: {
    name?: string;
    threshold?: number;
    checkpoint?: string;
  };
};

const BAR_COLORS = [
  'bg-red-500',
  'bg-orange-500',
  'bg-yellow-500',
  'bg-green-500',
  'bg-blue-500',
  'bg-purple-500',
];

const EXPLANATION_WINDOW_SEC = 10;
const TEMPORAL_TOKEN_COUNT = 50;
const SEGMENT_DURATION_SEC = EXPLANATION_WINDOW_SEC / TEMPORAL_TOKEN_COUNT;

export default function Explainability() {
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

  const patientCode = storedResult?.patient?.inputPatientCode ?? 'N/A';
  const analysis = storedResult?.analysis;
  const timeline = storedResult?.timeline ?? [];
  const explainability = storedResult?.explainability;
  const inferenceId = storedResult?.inference_id;

  const handleViewAnalysis = () => {
    navigate('/app/analysis');
  };

  const handleViewReport = () => {
    if (!inferenceId) {
      alert('Inference ID not found. Please re-run the analysis.');
      return;
    }
    navigate(`/app/reports?inference_id=${inferenceId}`);
  };

  const seizureDetected =
    analysis?.seizure_detected ??
    (analysis?.overall_prediction === 'seizure_activity_detected');

  const confidenceLevel = (analysis?.confidence_level ?? '').toLowerCase();

  const isLowConfidencePositive =
    seizureDetected && confidenceLevel.includes('low') && !confidenceLevel.includes('very high');

  const displayOutcome = !seizureDetected
    ? 'No Positive Windows Detected'
    : isLowConfidencePositive
    ? 'Low-Confidence Positive — Review Needed'
    : 'Positive Windows Detected';

  const outcomeTone = !seizureDetected
    ? 'negative'
    : isLowConfidencePositive
    ? 'warning'
    : 'positive';

  const normalizedTopChannels = useMemo(() => {
    const rawChannels = explainability?.top_channels ?? [];

    if (!Array.isArray(rawChannels)) return [];

    return rawChannels.map((item, index) => {
      if (typeof item === 'string') {
        return {
          channel: item,
          importance: Math.max(100 - index * 10, 40),
          color: BAR_COLORS[index % BAR_COLORS.length],
        };
      }

      const importanceValue =
        typeof item.importance === 'number'
          ? item.importance
          : typeof item.score === 'number'
          ? item.score * 100
          : Math.max(100 - index * 10, 40);

      return {
        channel: item.channel,
        importance: Math.max(0, Math.min(100, Math.round(importanceValue))),
        color: BAR_COLORS[index % BAR_COLORS.length],
      };
    });
  }, [explainability]);

  const normalizedTemporalAttention = useMemo(() => {
    const rawAttention = explainability?.temporal_attention ?? [];

    if (!Array.isArray(rawAttention)) return [];

    return rawAttention.map((item) => {
      const segStart = item.window_index * SEGMENT_DURATION_SEC;
      const segEnd = segStart + SEGMENT_DURATION_SEC;

      return {
        segmentIndex: item.window_index,
        label: `Segment ${item.window_index + 1}`,
        timeRange: `${segStart.toFixed(1)}s - ${segEnd.toFixed(1)}s`,
        attention: Math.max(0, Math.min(100, Math.round(item.attention * 100))),
        rawAttention: item.attention,
      };
    });
  }, [explainability]);

  const strongestAttentionWindow = useMemo(() => {
    if (!normalizedTemporalAttention.length) return null;

    return normalizedTemporalAttention.reduce((best, current) =>
      current.attention > best.attention ? current : best
    );
  }, [normalizedTemporalAttention]);

  const topTemporalSegments = useMemo(() => {
    return [...normalizedTemporalAttention]
      .sort((a, b) => b.attention - a.attention)
      .slice(0, 5);
  }, [normalizedTemporalAttention]);

  const strongestEdge = useMemo(() => {
    const edges = explainability?.gat_edges ?? [];
    if (!Array.isArray(edges) || edges.length === 0) return null;

    return edges.reduce((best, current) =>
      current.weight > best.weight ? current : best
    );
  }, [explainability]);

  const topGatEdges = useMemo(() => {
    const edges = explainability?.gat_edges ?? [];
    if (!Array.isArray(edges) || edges.length === 0) return [];

    const sorted = [...edges].sort((a, b) => b.weight - a.weight).slice(0, 8);
    const maxWeight = sorted[0]?.weight ?? 1;

    return sorted.map((edge) => ({
      ...edge,
      weightPercent: maxWeight > 0 ? Math.round((edge.weight / maxWeight) * 100) : 0,
    }));
  }, [explainability]);

  const timelineSummary = useMemo(() => {
    const positives = timeline.filter((item) => item.predicted_label === 1);

    if (positives.length === 0) return null;

    const startMin = (positives[0].start_sec / 60).toFixed(2);
    const endMin = (positives[positives.length - 1].end_sec / 60).toFixed(2);

    return {
      startMin,
      endMin,
      positiveCount: positives.length,
    };
  }, [timeline]);

  const temporalSummarySentence = useMemo(() => {
    if (!strongestAttentionWindow) return null;
    return `The model focused most strongly on ${strongestAttentionWindow.timeRange} within the selected explanation window.`;
  }, [strongestAttentionWindow]);

  const plainLanguageSummary = useMemo(() => {
    const topChannel = normalizedTopChannels[0]?.channel;
    const topSegment = strongestAttentionWindow?.timeRange;
    const topConnection = strongestEdge
      ? `${strongestEdge.source} → ${strongestEdge.target}`
      : null;

    if (!seizureDetected) {
      return 'The model did not identify seizure-positive windows in this result. The explainability outputs below show available influence patterns for review, but they do not indicate a confirmed seizure event.';
    }

    if (isLowConfidencePositive) {
      return `The model flagged possible seizure-like activity, but confidence is limited. ${
        topChannel ? `The most influential channel was ${topChannel}. ` : ''
      }${
        topSegment ? `The strongest temporal focus was around ${topSegment}. ` : ''
      }${
        topConnection ? `The strongest channel interaction was ${topConnection}.` : ''
      }`;
    }

    return `The model identified strong seizure-related activity in this result. ${
      topChannel ? `The most influential channel was ${topChannel}. ` : ''
    }${
      topSegment ? `The strongest temporal focus was around ${topSegment}. ` : ''
    }${
      topConnection ? `The strongest channel interaction was ${topConnection}.` : ''
    }`;
  }, [
    seizureDetected,
    isLowConfidencePositive,
    normalizedTopChannels,
    strongestAttentionWindow,
    strongestEdge,
  ]);

  if (!storedResult || !analysis) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Explainable AI Analysis
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            No completed explainability result was found.
          </p>
        </div>

        <Card className="dark:bg-slate-800 dark:border-gray-700">
          <CardContent className="p-6">
            <p className="text-gray-700 dark:text-gray-300">
              Please run EEG analysis first, then return to this page.
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
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Explainable AI Analysis
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Understanding how the model produced its prediction for Patient {patientCode}
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          <Button
            onClick={handleViewAnalysis}
            className="bg-blue-600 text-white hover:bg-blue-700 dark:bg-teal-600 dark:hover:bg-teal-700"
          >
            <Activity className="mr-2 h-4 w-4" />
            View Analysis
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

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.35fr_0.9fr]">
        <Card className="overflow-hidden border border-blue-300/70 bg-[linear-gradient(135deg,#ECF7FF_0%,#DCEFFF_50%,#CFE6FA_100%)] shadow-2xl dark:border-blue-400/30 dark:bg-[linear-gradient(135deg,#4A8A9E_0%,#3A5A78_55%,#2A4A5E_100%)]">
          <CardContent className="p-0">
            <div className="relative overflow-hidden px-7 py-7">
              <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-white/40 blur-2xl dark:bg-white/10" />
              <div className="absolute -bottom-10 -left-10 h-40 w-40 rounded-full bg-sky-200/40 blur-2xl dark:bg-black/10" />

              <div className="relative space-y-6">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="space-y-2">
                    <div className="inline-flex items-center gap-2 rounded-full border border-blue-300/70 bg-white/65 px-3 py-1 text-sm font-medium text-sky-900 dark:border-blue-400/30 dark:bg-[rgba(42,74,94,0.35)] dark:text-white/90">
                      <Brain className="h-4 w-4" />
                      Quick Clinical Interpretation
                    </div>

                    <h2 className="max-w-3xl text-3xl font-bold leading-tight text-slate-900 md:text-4xl dark:text-white">
                      {seizureDetected
                        ? isLowConfidencePositive
                          ? 'Possible seizure activity influenced the AI result'
                          : 'Strong seizure-related activity influenced the AI result'
                        : 'No seizure activity was highlighted by the AI'}
                    </h2>

                    <p className="max-w-2xl text-sm leading-6 text-slate-700 md:text-base dark:text-white/85">
                      {seizureDetected
                        ? isLowConfidencePositive
                          ? 'The model flagged seizure-like activity, but confidence is limited. The strongest evidence came from the highlighted time segment and channel interaction shown below.'
                          : 'The model focused most strongly on the highlighted EEG features below when producing this result.'
                        : 'The model did not identify seizure-positive windows. The sections below show which EEG features were reviewed and how the model distributed its attention.'}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-blue-300/70 bg-white/60 px-4 py-3 text-center backdrop-blur-sm dark:border-blue-400/30 dark:bg-[rgba(42,74,94,0.35)]">
                    <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-white/75">
                      Model Outcome
                    </p>
                    <p className="mt-1 text-lg font-bold text-slate-900 dark:text-white">
                      {displayOutcome}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                  <div className="rounded-2xl border border-blue-300/70 bg-white/60 p-5 backdrop-blur-sm dark:border-blue-400/20 dark:bg-[rgba(42,74,94,0.42)]">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-white/60">
                      Most Important Channel
                    </p>
                    <p className="mt-2 text-2xl font-bold text-slate-900 dark:text-white">
                      {normalizedTopChannels[0]?.channel ?? 'Pending'}
                    </p>
                    <p className="mt-1 text-sm text-slate-600 dark:text-white/70">
                      Where the model looked most strongly
                    </p>
                  </div>

                  <div className="rounded-2xl border border-blue-300/70 bg-white/60 p-5 backdrop-blur-sm dark:border-blue-400/20 dark:bg-[rgba(42,74,94,0.42)]">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-white/60">
                      Strongest Time Focus
                    </p>
                    <p className="mt-2 text-2xl font-bold text-slate-900 dark:text-white">
                      {strongestAttentionWindow?.timeRange ?? 'N/A'}
                    </p>
                    <p className="mt-1 text-sm text-slate-600 dark:text-white/70">
                      When the strongest influence happened
                    </p>
                  </div>

                  <div className="rounded-2xl border border-blue-300/70 bg-white/60 p-5 backdrop-blur-sm dark:border-blue-400/20 dark:bg-[rgba(42,74,94,0.42)]">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-white/60">
                      Strongest Connection
                    </p>
                    <p className="mt-2 text-2xl font-bold text-slate-900 dark:text-white">
                      {strongestEdge
                        ? `${strongestEdge.source} → ${strongestEdge.target}`
                        : 'N/A'}
                    </p>
                    <p className="mt-1 text-sm text-slate-600 dark:text-white/70">
                      Channel relationship with highest influence
                    </p>
                  </div>
                </div>

                <div className="rounded-2xl border border-blue-300/70 bg-white/60 px-5 py-4 text-sm text-slate-700 backdrop-blur-sm dark:border-blue-400/20 dark:bg-[rgba(42,74,94,0.50)] dark:text-white/90">
                  <span className="font-semibold text-slate-900 dark:text-white">Simple meaning:</span>{' '}
                  These highlighted EEG parts contributed most to the AI decision. They help explain
                  the model’s reasoning, but they do not confirm diagnosis on their own.
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border border-blue-300/70 bg-gradient-to-b from-slate-50 to-blue-50 shadow-xl dark:border-blue-400/30 dark:from-slate-900 dark:to-blue-950/60">
          <CardContent className="p-6">
            <div className="space-y-5">
              <div className="space-y-2">
                <div className="inline-flex items-center gap-2 rounded-full border border-blue-300 bg-blue-100 px-3 py-1 text-sm font-medium text-blue-800 dark:border-blue-400/20 dark:bg-blue-500/10 dark:text-blue-200">
                  <Info className="h-4 w-4" />
                  How to Read This Page
                </div>

                <p className="text-sm leading-6 text-slate-700 dark:text-slate-300">
                  Use this page as a quick guide to understand <span className="font-semibold text-slate-900 dark:text-white">why</span> the AI
                  gave its result.
                </p>
              </div>

              <div className="space-y-3">
                <div className="rounded-xl border border-slate-200 bg-white/80 p-4 dark:border-white/10 dark:bg-white/5">
                  <p className="text-sm font-semibold text-slate-900 dark:text-white">Channels</p>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                    Show <span className="font-medium text-blue-700 dark:text-blue-200">where</span> the model focused most strongly in the EEG.
                  </p>
                </div>

                <div className="rounded-xl border border-slate-200 bg-white/80 p-4 dark:border-white/10 dark:bg-white/5">
                  <p className="text-sm font-semibold text-slate-900 dark:text-white">Time Segments</p>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                    Show <span className="font-medium text-blue-700 dark:text-blue-200">when</span> the strongest influence happened.
                  </p>
                </div>

                <div className="rounded-xl border border-slate-200 bg-white/80 p-4 dark:border-white/10 dark:bg-white/5">
                  <p className="text-sm font-semibold text-slate-900 dark:text-white">Connections</p>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                    Show which EEG channel relationships mattered most to the model.
                  </p>
                </div>
              </div>

              <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 dark:border-amber-400/30 dark:bg-amber-500/10">
                <p className="text-sm font-semibold text-amber-800 dark:text-amber-200">
                  Important
                </p>
                <p className="mt-1 text-sm leading-6 text-amber-900/80 dark:text-amber-100/90">
                  These visuals explain model reasoning only. They should support EEG review, not replace
                  expert clinical judgment.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="dark:bg-slate-800 dark:border-gray-700">
        <CardHeader>
          <CardTitle className="text-gray-900 dark:text-white flex items-center gap-2">
            <Zap className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />
            Quick Clinical Interpretation
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            This section summarizes the most influential components returned by the backend explainability pipeline.
          </p>

          <div className="bg-gray-50 dark:bg-slate-900 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700 dark:text-gray-300">Prediction Outcome</span>
              <Badge
                className={
                  outcomeTone === 'negative'
                    ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200'
                    : outcomeTone === 'warning'
                    ? 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200'
                    : 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200'
                }
              >
                {displayOutcome}
              </Badge>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700 dark:text-gray-300">Saliency Output</span>
              <span className="text-sm font-semibold text-gray-900 dark:text-white">
                {explainability?.saliency_path || explainability?.saliency_available
                  ? 'Available'
                  : 'Not provided in current response'}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700 dark:text-gray-300">Strongest Temporal Focus</span>
              <span className="text-sm font-semibold text-gray-900 dark:text-white">
                {strongestAttentionWindow
                  ? `${strongestAttentionWindow.timeRange} (${strongestAttentionWindow.attention}%)`
                  : explainability?.temporal_attention_path
                  ? 'Available in backend artifact'
                  : 'N/A'}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700 dark:text-gray-300">Strongest GAT Interaction</span>
              <span className="text-sm font-semibold text-gray-900 dark:text-white">
                {strongestEdge
                  ? `${strongestEdge.source} → ${strongestEdge.target}`
                  : explainability?.gat_attention_path
                  ? 'Available in backend artifact'
                  : 'N/A'}
              </span>
            </div>
          </div>

          <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              {explainability?.summary_text ??
                (!seizureDetected
                  ? 'No seizure-positive windows were identified in the current result. The interface is showing available explainability outputs for review.'
                  : isLowConfidencePositive
                  ? 'The model produced a low-confidence positive result. These explainability outputs should be reviewed carefully alongside the raw EEG and clinical context.'
                  : 'The model produced a strong positive result. These explainability outputs summarize the channels, temporal segments, and graph interactions that most influenced the decision.')}
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="dark:bg-slate-800 dark:border-gray-700">
          <CardHeader>
            <CardTitle className="text-gray-900 dark:text-white flex items-center gap-2">
              <Brain className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              Top Important EEG Channels
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {normalizedTopChannels.length === 0 ? (
              <p className="text-sm text-gray-600 dark:text-gray-400">
                No channel-importance data is available yet from the backend.
              </p>
            ) : (
              normalizedTopChannels.map((channel, idx) => (
                <div key={`${channel.channel}-${idx}`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className="font-mono dark:border-gray-600">
                        {channel.channel}
                      </Badge>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Rank {idx + 1}
                      </span>
                    </div>
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {channel.importance}%
                    </span>
                  </div>
                  <div className="space-y-2">
                    <Progress value={channel.importance} className="h-2" />
                    <div className={`h-1 rounded ${channel.color}`} />
                  </div>
                </div>
              ))
            )}

            <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg">
              <p className="text-sm text-purple-800 dark:text-purple-200">
                {normalizedTopChannels.length > 0
                  ? `The highest-ranked channel in the current explainability output is ${normalizedTopChannels[0].channel}.`
                  : 'Top-channel interpretation will appear here once channel-importance values are returned by the backend.'}
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="dark:bg-slate-800 dark:border-gray-700">
          <CardHeader>
            <CardTitle className="text-gray-900 dark:text-white flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-teal-600 dark:text-teal-400" />
              Temporal Importance Map
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            {normalizedTemporalAttention.length === 0 ? (
              <p className="text-sm text-gray-600 dark:text-gray-400">
                No temporal-attention values are available yet from the backend.
              </p>
            ) : (
              <>
                <div className="p-4 rounded-lg bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-800">
                  <p className="text-sm text-teal-800 dark:text-teal-200">
                    {temporalSummarySentence}
                  </p>
                </div>

                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                    Top 5 most influential temporal segments in the selected explanation window.
                  </p>

                  <div className="space-y-4">
                    {topTemporalSegments.map((segment, idx) => (
                      <div key={`${segment.segmentIndex}-${idx}`}>
                        <div className="flex items-center justify-between mb-2">
                          <div>
                            <p className="text-sm font-medium text-gray-900 dark:text-white">
                              {segment.label}
                            </p>
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                              {segment.timeRange}
                            </p>
                          </div>
                          <span className="text-sm font-semibold text-gray-900 dark:text-white">
                            {segment.attention}%
                          </span>
                        </div>
                        <Progress value={segment.attention} className="h-2" />
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                    Full temporal profile across all 50 segments.
                  </p>
                  <div className="grid grid-cols-10 gap-2">
                    {normalizedTemporalAttention.map((segment) => (
                      <div
                        key={segment.segmentIndex}
                        className="space-y-1"
                        title={`${segment.label} • ${segment.timeRange} • ${segment.attention}%`}
                      >
                        <div className="h-16 bg-gray-100 dark:bg-slate-900 rounded flex items-end overflow-hidden">
                          <div
                            className="w-full bg-teal-500 rounded-t"
                            style={{ height: `${Math.max(segment.attention, 4)}%` }}
                          />
                        </div>
                        <p className="text-[10px] text-center text-gray-500 dark:text-gray-400">
                          {segment.segmentIndex + 1}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            <div className="p-4 bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-800 rounded-lg">
              <p className="text-sm text-teal-800 dark:text-teal-200">
                {strongestAttentionWindow
                  ? `Peak temporal importance was observed around ${strongestAttentionWindow.timeRange}.`
                  : explainability?.temporal_attention_path
                  ? 'Temporal-attention artifact exists in the backend, but structured values were not returned in the API response.'
                  : 'Temporal attention interpretation will appear here once attention weights are returned by the backend.'}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="dark:bg-slate-800 dark:border-gray-700">
        <CardHeader>
          <CardTitle className="text-gray-900 dark:text-white flex items-center gap-2">
            <Network className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
            Top Graph Attention Connections
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {topGatEdges.length === 0 ? (
            <p className="text-sm text-gray-600 dark:text-gray-400">
              No graph-attention edge data is available yet from the backend.
            </p>
          ) : (
            topGatEdges.map((edge, idx) => (
              <div key={`${edge.source}-${edge.target}-${idx}`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <Badge variant="outline" className="font-mono dark:border-gray-600">
                      {edge.source}
                    </Badge>
                    <span className="text-gray-500 dark:text-gray-400">→</span>
                    <Badge variant="outline" className="font-mono dark:border-gray-600">
                      {edge.target}
                    </Badge>
                  </div>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {edge.weight.toFixed(3)}
                  </span>
                </div>
                <Progress value={edge.weightPercent} className="h-2" />
              </div>
            ))
          )}

          <div className="mt-6 p-4 bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-lg">
            <p className="text-sm text-indigo-800 dark:text-indigo-200">
              {topGatEdges.length > 0
                ? `The strongest graph-based interaction in the current result is ${topGatEdges[0].source} → ${topGatEdges[0].target}.`
                : 'Top graph-attention connections will appear here once GAT edge weights are returned by the backend.'}
            </p>
          </div>
        </CardContent>
      </Card>

      <Card className="dark:bg-slate-800 dark:border-gray-700">
        <CardHeader>
          <CardTitle className="text-gray-900 dark:text-white">
            Clinical Decision Support Notes
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-4 text-gray-700 dark:text-gray-300">
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Key Findings</h4>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>
                  {timelineSummary
                    ? `Model-positive windows were observed approximately between ${timelineSummary.startMin} and ${timelineSummary.endMin} minutes.`
                    : 'No positive seizure windows were identified in the currently loaded result.'}
                </li>
                <li>
                  {normalizedTopChannels.length > 0
                    ? `Most influential channel: ${normalizedTopChannels[0].channel}.`
                    : 'Channel-level influence data is not currently available.'}
                </li>
                <li>
                  {strongestEdge
                    ? `Strongest graph interaction: ${strongestEdge.source} to ${strongestEdge.target} with weight ${strongestEdge.weight.toFixed(3)}.`
                    : explainability?.gat_attention_path
                    ? 'Graph-attention artifact exists in the backend, but structured edge values were not returned in the API response.'
                    : 'Graph-attention edge summaries are not currently available.'}
                </li>
                <li>
                  {strongestAttentionWindow
                    ? `Peak temporal importance occurred around ${strongestAttentionWindow.timeRange}.`
                    : explainability?.temporal_attention_path
                    ? 'Temporal-attention artifact exists in the backend, but structured values were not returned in the API response.'
                    : 'Temporal focus summary is not currently available.'}
                </li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                Interpretation Notes
              </h4>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>
                  These explanations reflect which inputs most influenced the model, not a direct clinical diagnosis.
                </li>
                <li>
                  Explainability outputs should be reviewed together with raw EEG signals and the broader patient context.
                </li>
                <li>
                  Low-confidence positive outputs should be treated as review-recommended alerts rather than definitive seizure confirmation.
                </li>
                <li>
                  Temporal attention is summarized from the representative explanation window to improve readability in the interface.
                </li>
              </ul>
            </div>

            <div className="p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
              <p className="text-sm text-amber-900 dark:text-amber-200">
                <strong>Research Prototype Recommendation:</strong> These AI explanations are intended to support
                interpretation in this final-year project prototype. They should assist, not replace, expert EEG review
                and clinical judgment.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}