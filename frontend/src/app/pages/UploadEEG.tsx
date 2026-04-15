import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileText, CheckCircle, AlertCircle, BarChart3, FileOutput } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Button } from '../components/ui/button';
import { Progress } from '../components/ui/progress';

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

type PreviewChannel = {
  name: string;
  values: number[];
};

type PreviewData = {
  sampling_rate: number;
  duration_sec: number;
  channels: PreviewChannel[];
};

export default function UploadEEG() {
  const navigate = useNavigate();

  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [patientId, setPatientId] = useState('');
  const [recordingDate, setRecordingDate] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [preview, setPreview] = useState<PreviewData | null>(null);
  const [latestInferenceId, setLatestInferenceId] = useState<number | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (!droppedFile.name.toLowerCase().endsWith('.edf')) {
        setErrorMessage('Only EDF files are supported.');
        return;
      }
      setErrorMessage('');
      setFile(droppedFile);
      setPreview(null);
      setUploadComplete(false);
      setLatestInferenceId(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (!selectedFile.name.toLowerCase().endsWith('.edf')) {
        setErrorMessage('Only EDF files are supported.');
        return;
      }
      setErrorMessage('');
      setFile(selectedFile);
      setPreview(null);
      setUploadComplete(false);
      setLatestInferenceId(null);
    }
  };

  const buildWavePath = (
    values: number[],
    width: number,
    yOffset: number,
    amplitude: number
  ) => {
    if (!values || values.length === 0) return '';

    const maxAbs = Math.max(...values.map((v) => Math.abs(v)), 1);
    const stepX = width / Math.max(values.length - 1, 1);

    return values
      .map((v, i) => {
        const x = i * stepX;
        const y = yOffset - (v / maxAbs) * amplitude;
        return `${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(' ');
  };

  const handleUpload = async () => {
    if (!file || !patientId || !recordingDate) {
      setErrorMessage('Please fill all fields.');
      return;
    }

    setUploading(true);
    setUploadProgress(10);
    setErrorMessage('');
    setUploadComplete(false);
    setLatestInferenceId(null);

    try {
      let backendPatientId: number | null = null;

      const patientsResponse = await fetch(`${API_BASE_URL}/patients`);
      if (!patientsResponse.ok) {
        throw new Error('Failed to load patients');
      }

      const patients = await patientsResponse.json();

      const existingPatient = Array.isArray(patients)
        ? patients.find((p: any) => String(p.patient_code) === String(patientId))
        : null;

      if (existingPatient) {
        backendPatientId = existingPatient.id;
      } else {
        const createPatientResponse = await fetch(`${API_BASE_URL}/patients`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            patient_code: patientId,
            display_name: `Patient ${patientId}`,
            notes: `Recording date: ${recordingDate}`,
          }),
        });

        if (!createPatientResponse.ok) {
          const createErr = await createPatientResponse.json().catch(() => null);
          throw new Error(createErr?.detail || 'Patient creation failed');
        }

        const createdPatient = await createPatientResponse.json();
        backendPatientId = createdPatient.id;
      }

      setUploadProgress(35);

      const formData = new FormData();
      formData.append('patient_id', String(backendPatientId));
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/eeg-records/upload-and-analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => null);
        throw new Error(err?.detail || 'Upload and analysis failed');
      }

      const responseData = await response.json();

      const uploadData = responseData.record;
      const inferenceData = responseData.result;

      setPreview(inferenceData.preview ?? null);
      setLatestInferenceId(inferenceData.inference_id ?? null);
      setUploadProgress(90);

      localStorage.setItem(
        'latestInferenceResult',
        JSON.stringify({
          inference_id: inferenceData.inference_id,
          patient: {
            inputPatientCode: patientId,
            backendPatientId,
            recordingDate,
          },
          upload: uploadData,
          analysis: inferenceData.analysis,
          timeline: inferenceData.timeline,
          preview: inferenceData.preview,
          explainability: inferenceData.explainability,
          model: inferenceData.model,
        })
      );

      setUploadProgress(100);
      setUploadComplete(true);
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Something went wrong';
      setErrorMessage(msg);
      setUploadProgress(0);
      setUploadComplete(false);
      setLatestInferenceId(null);
    } finally {
      setUploading(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setPatientId('');
    setRecordingDate('');
    setUploadProgress(0);
    setUploadComplete(false);
    setErrorMessage('');
    setPreview(null);
    setLatestInferenceId(null);
  };

  const goToAnalysis = () => {
    navigate('/app/analysis');
  };

  const goToReport = () => {
    if (!latestInferenceId) {
      setErrorMessage('Report cannot be opened because inference_id is missing.');
      return;
    }
    navigate(`/app/reports?inference_id=${latestInferenceId}`);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Upload EEG Recording</h1>
        <p className="mt-1 text-gray-600 dark:text-gray-400">
          Upload EDF format EEG recordings for AI-powered seizure detection
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card className="dark:border-gray-700 dark:bg-slate-800">
          <CardHeader>
            <CardTitle className="text-gray-900 dark:text-white">EEG File Upload</CardTitle>
          </CardHeader>

          <CardContent className="space-y-6">
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
                dragActive
                  ? 'border-blue-500 bg-blue-50 dark:border-teal-400 dark:bg-blue-900/20'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
            >
              {file ? (
                <div className="space-y-3">
                  <CheckCircle className="mx-auto h-12 w-12 text-green-600 dark:text-green-400" />
                  <div>
                    <p className="font-semibold text-gray-900 dark:text-white">{file.name}</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setFile(null);
                      setPreview(null);
                      setUploadComplete(false);
                      setLatestInferenceId(null);
                    }}
                    className="dark:border-gray-600"
                    disabled={uploading}
                  >
                    Change File
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  <Upload className="mx-auto h-12 w-12 text-gray-400" />
                  <div>
                    <p className="font-semibold text-gray-900 dark:text-white">
                      Drag and drop your EEG file here
                    </p>
                    <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">or</p>
                  </div>

                  <label htmlFor="file-upload">
                    <Button variant="outline" asChild className="dark:border-gray-600">
                      <span className="cursor-pointer">Browse Files</span>
                    </Button>
                  </label>

                  <input
                    id="file-upload"
                    type="file"
                    accept=".edf"
                    onChange={handleFileChange}
                    className="hidden"
                    disabled={uploading}
                  />

                  <p className="mt-2 text-xs text-gray-500 dark:text-gray-500">
                    Supported format: EDF (European Data Format)
                  </p>
                </div>
              )}
            </div>

            <div className="space-y-4">
              <div>
                <Label htmlFor="patientId" className="text-gray-700 dark:text-gray-300">
                  Patient ID
                </Label>
                <Input
                  id="patientId"
                  type="text"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  placeholder="PT-1234"
                  className="mt-1 dark:border-gray-700 dark:bg-slate-900"
                  disabled={uploading}
                />
              </div>

              <div>
                <Label htmlFor="recordingDate" className="text-gray-700 dark:text-gray-300">
                  Recording Date
                </Label>
                <Input
                  id="recordingDate"
                  type="datetime-local"
                  value={recordingDate}
                  onChange={(e) => setRecordingDate(e.target.value)}
                  className="mt-1 dark:border-gray-700 dark:bg-slate-900"
                  disabled={uploading}
                />
              </div>
            </div>

            {uploading && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">Uploading and analyzing...</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {uploadProgress}%
                  </span>
                </div>
                <Progress value={uploadProgress} className="h-2" />
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Processing may take some time for long EDF recordings.
                </p>
              </div>
            )}

            {errorMessage && (
              <div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-900/20">
                <div className="flex items-center gap-2 text-red-800 dark:text-red-200">
                  <AlertCircle className="h-5 w-5" />
                  <p className="font-semibold">Upload Failed</p>
                </div>
                <p className="mt-1 text-sm text-red-700 dark:text-red-300">{errorMessage}</p>
              </div>
            )}

            {uploadComplete && (
              <div className="rounded-lg border border-green-200 bg-green-50 p-4 dark:border-green-800 dark:bg-green-900/20">
                <div className="flex items-center gap-2 text-green-800 dark:text-green-200">
                  <CheckCircle className="h-5 w-5" />
                  <p className="font-semibold">Analysis Complete!</p>
                </div>
                <p className="mt-1 text-sm text-green-700 dark:text-green-300">
                  EEG recording was processed successfully. You can now review the preview below and open the full analysis page when ready.
                </p>
              </div>
            )}

            <div className="flex flex-wrap gap-3">
              {uploadComplete ? (
                <>
                  <Button
                    onClick={goToAnalysis}
                    className="min-w-[180px] flex-1 bg-blue-600 text-white hover:bg-blue-700 dark:bg-teal-600 dark:hover:bg-teal-700"
                  >
                    <BarChart3 className="mr-2 h-4 w-4" />
                    View Analysis
                  </Button>

                  <Button
                    onClick={goToReport}
                    variant="outline"
                    className="min-w-[180px] flex-1 dark:border-gray-600"
                  >
                    <FileOutput className="mr-2 h-4 w-4" />
                    View Report
                  </Button>

                  <Button
                    variant="outline"
                    onClick={resetForm}
                    className="min-w-[180px] flex-1 dark:border-gray-600"
                  >
                    Upload Another File
                  </Button>
                </>
              ) : (
                <>
                  <Button
                    onClick={handleUpload}
                    disabled={!file || !patientId || !recordingDate || uploading}
                    className="flex-1 bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 dark:bg-teal-600 dark:hover:bg-teal-700"
                  >
                    {uploading ? 'Processing...' : 'Upload & Analyze'}
                  </Button>

                  <Button
                    variant="outline"
                    onClick={resetForm}
                    disabled={uploading}
                    className="dark:border-gray-600"
                  >
                    Reset
                  </Button>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card className="dark:border-gray-700 dark:bg-slate-800">
            <CardHeader>
              <CardTitle className="text-gray-900 dark:text-white">EEG Signal Preview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative h-64 overflow-hidden rounded-lg bg-gray-900 p-4 dark:bg-black">
                {preview && preview.channels.length > 0 ? (
                  <svg className="h-full w-full" viewBox="0 0 400 220" preserveAspectRatio="none">
                    {preview.channels.map((channel, idx) => {
                      const yOffset = 50 + idx * 60;
                      const colors = ['#10B981', '#3B82F6', '#14B8A6', '#A855F7'];
                      return (
                        <g key={channel.name}>
                          <text
                            x="8"
                            y={yOffset - 18}
                            fill="#CBD5E1"
                            fontSize="10"
                            className="select-none"
                          >
                            {channel.name}
                          </text>
                          <path
                            d={buildWavePath(channel.values, 400, yOffset, 16)}
                            stroke={colors[idx % colors.length]}
                            strokeWidth="1.8"
                            fill="none"
                            opacity="0.9"
                          />
                        </g>
                      );
                    })}
                  </svg>
                ) : (
                  <div className="flex h-full w-full items-center justify-center text-center">
                    <div>
                      <p className="text-sm font-medium text-gray-300">
                        EEG preview will appear after analysis completes
                      </p>
                      <p className="mt-2 text-xs text-gray-500">
                        Showing the first few seconds from selected bipolar channels
                      </p>
                    </div>
                  </div>
                )}
              </div>

              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
                {preview
                  ? `Previewing ${preview.channels.length} channel(s) over the first ${preview.duration_sec} seconds.`
                  : 'A lightweight waveform preview will be generated from the processed EEG signal.'}
              </p>
            </CardContent>
          </Card>

          <Card className="dark:border-gray-700 dark:bg-slate-800">
            <CardHeader>
              <CardTitle className="text-gray-900 dark:text-white">Upload Guidelines</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start gap-3">
                <FileText className="mt-0.5 h-5 w-5 text-blue-600 dark:text-teal-400" />
                <div>
                  <p className="text-sm font-semibold text-gray-900 dark:text-white">File Format</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Only EDF (European Data Format) files are supported
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <AlertCircle className="mt-0.5 h-5 w-5 text-blue-600 dark:text-teal-400" />
                <div>
                  <p className="text-sm font-semibold text-gray-900 dark:text-white">
                    Recording Quality
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Ensure recordings are of good quality with minimal artifacts
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <CheckCircle className="mt-0.5 h-5 w-5 text-blue-600 dark:text-teal-400" />
                <div>
                  <p className="text-sm font-semibold text-gray-900 dark:text-white">
                    Research Prototype
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Uploaded data is used within this project prototype for EEG analysis and report generation
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}