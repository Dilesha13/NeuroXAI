import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Filter, Download, Eye, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table';
import { API_BASE_URL } from '../../config';

type PatientRecord = {
  record_id: number;
  patient_id: number;
  patient_code: string;
  patient_name?: string | null;
  recording_date?: string | null;
  duration_minutes?: number | null;
  duration_label: string;
  status: string;
  confidence?: number | null;
  result: string;
  record_status: string;
  report_id?: number | null;
  download_url?: string | null;
  inference_id?: number | null;
  filename?: string | null;
};

function formatDate(value?: string | null): string {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function getBadgeClass(status: string): string {
  if (status === 'Seizure Detected') {
    return 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200';
  }
  if (
    status === 'Pending' ||
    status === 'Review Needed' ||
    status === 'Possible Seizure Activity — Review Needed'
  ) {
    return 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200';
  }
  return 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200';
}

function exportCsv(records: PatientRecord[]) {
  const rows = [
    ['Patient ID', 'Patient Name', 'Recording Date', 'Duration', 'Status', 'Confidence', 'Result', 'File Name'],
    ...records.map((record) => [
      record.patient_code,
      record.patient_name ?? '',
      formatDate(record.recording_date),
      record.duration_label,
      record.status,
      record.confidence != null ? `${record.confidence.toFixed(1)}%` : '—',
      record.result,
      record.filename ?? '',
    ]),
  ];

  const csv = rows
    .map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(','))
    .join('\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'patient-records.csv';
  link.click();
  URL.revokeObjectURL(url);
}


export default function PatientRecords() {
  const [records, setRecords] = useState<PatientRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const navigate = useNavigate();

  const itemsPerPage = 10;

    const handleViewReport = (record: PatientRecord) => {
      if (!record.inference_id) return;
      navigate(`/app/reports?inference_id=${record.inference_id}`);
  };

  const handleDownloadReport = async (record: PatientRecord) => {
      if (!record.inference_id) return;

      try {
        const token = localStorage.getItem('neuroxai-token');

        const response = await fetch(`${API_BASE_URL}/reports/${record.inference_id}/generate`, {
          method: 'POST',
          headers: token
            ? {
                Authorization: `Bearer ${token}`,
              }
            : {},
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Failed to generate report');
        }

        const normalizedBase = API_BASE_URL.replace(/\/api\/v1\/?$/, '');
        const fullDownloadUrl = `${normalizedBase}${data.download_url}`;
        window.open(fullDownloadUrl, '_blank');
      } catch (err) {
        console.error(err);
        alert(err instanceof Error ? err.message : 'Failed to download report');
      }
    };

  useEffect(() => {
    let isMounted = true;

    async function loadRecords() {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`${API_BASE_URL}/patients/records`);
        if (!response.ok) {
          throw new Error('Failed to load patient records');
        }

        const data: PatientRecord[] = await response.json();

        if (isMounted) {
          setRecords(data);
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

    loadRecords();

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, filterStatus]);

  const filteredData = useMemo(() => {
    return records.filter((record) => {
      const search = searchQuery.trim().toLowerCase();

      const matchesSearch =
        search.length === 0 ||
        record.patient_code.toLowerCase().includes(search) ||
        (record.patient_name ?? '').toLowerCase().includes(search) ||
        String(record.record_id).includes(search);

    const matchesFilter =
      filterStatus === 'all' ||
      (filterStatus === 'seizure' && record.status === 'Seizure Detected') ||
      (filterStatus === 'normal' && record.result === 'Normal') ||
      (filterStatus === 'pending' &&
        (record.status === 'Review Needed' ||
          record.status === 'Possible Seizure Activity — Review Needed' ||
          record.result === 'Pending' ||
          record.result === 'Needs Review'));

      return matchesSearch && matchesFilter;
    });
  }, [records, searchQuery, filterStatus]);

  const totalPages = Math.max(1, Math.ceil(filteredData.length / itemsPerPage));
  const paginatedData = filteredData.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Patient Records</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Comprehensive EEG analysis history and patient data
        </p>
      </div>

      <Card className="dark:bg-slate-800 dark:border-gray-700">
        <CardContent className="p-6">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                type="text"
                placeholder="Search by Patient ID or name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 dark:bg-slate-900 dark:border-gray-700"
              />
            </div>

            <div className="w-full sm:w-52">
              <Select value={filterStatus} onValueChange={setFilterStatus}>
                <SelectTrigger className="dark:bg-slate-900 dark:border-gray-700">
                  <Filter className="h-4 w-4 mr-2" />
                  <SelectValue placeholder="Filter by status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Records</SelectItem>
                  <SelectItem value="seizure">Seizure Detected</SelectItem>
                  <SelectItem value="normal">Normal</SelectItem>
                  <SelectItem value="pending">Review Needed</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button
              variant="outline"
              className="dark:border-gray-700"
              onClick={() => exportCsv(filteredData)}
              disabled={filteredData.length === 0}
            >
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        </CardContent>
      </Card>

      {loading ? (
        <Card className="dark:bg-slate-800 dark:border-gray-700">
          <CardContent className="p-12 flex items-center justify-center gap-3 text-gray-600 dark:text-gray-300">
            <Loader2 className="h-5 w-5 animate-spin" />
            Loading patient records...
          </CardContent>
        </Card>
      ) : error ? (
        <Card className="dark:bg-slate-800 dark:border-gray-700">
          <CardContent className="p-6 text-red-600 dark:text-red-300">{error}</CardContent>
        </Card>
      ) : (
        <>
          <Card className="dark:bg-slate-800 dark:border-gray-700 hidden md:block">
            <CardHeader>
              <CardTitle className="text-gray-900 dark:text-white">Analysis Records</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow className="dark:border-gray-700">
                    <TableHead className="text-gray-900 dark:text-gray-300">Patient ID</TableHead>
                    <TableHead className="text-gray-900 dark:text-gray-300">Patient Name</TableHead>
                    <TableHead className="text-gray-900 dark:text-gray-300">Recording Date</TableHead>
                    <TableHead className="text-gray-900 dark:text-gray-300">Duration</TableHead>
                    <TableHead className="text-gray-900 dark:text-gray-300">Status</TableHead>
                    <TableHead className="text-gray-900 dark:text-gray-300">Confidence</TableHead>
                    <TableHead className="text-gray-900 dark:text-gray-300">Result</TableHead>
                    <TableHead className="text-gray-900 dark:text-gray-300">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paginatedData.map((record) => {
                    

                    return (
                      <TableRow key={record.record_id} className="dark:border-gray-700">
                        <TableCell className="font-medium text-gray-900 dark:text-white">
                          {record.patient_code}
                        </TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-400">
                          {record.patient_name ?? '—'}
                        </TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-400">
                          {formatDate(record.recording_date)}
                        </TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-400">
                          {record.duration_label}
                        </TableCell>
                        <TableCell>
                          <Badge className={getBadgeClass(record.status)}>{record.status}</Badge>
                        </TableCell>
                        <TableCell className="font-medium text-gray-900 dark:text-white">
                          {record.confidence != null ? `${record.confidence.toFixed(1)}%` : '—'}
                        </TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-400">
                          {record.result}
                        </TableCell>
                        <TableCell>
                          <div className="flex gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="dark:hover:bg-slate-700"
                              onClick={() => handleViewReport(record)}
                              disabled={!record.inference_id}
                              title={record.inference_id ? 'View report preview' : 'Analysis not available yet'}
                            >
                              <Eye className="h-4 w-4" />
                            </Button>

                            <Button
                              variant="ghost"
                              size="sm"
                              className="dark:hover:bg-slate-700"
                              onClick={() => handleDownloadReport(record)}
                              disabled={!record.inference_id}
                              title={record.inference_id ? 'Generate and download report' : 'Analysis not available yet'}
                            >
                              <Download className="h-4 w-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <div className="md:hidden space-y-4">
            {paginatedData.map((record) => {
              

              return (
                <Card key={record.record_id} className="dark:bg-slate-800 dark:border-gray-700">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-3 gap-3">
                      <div>
                        <p className="font-semibold text-gray-900 dark:text-white">
                          {record.patient_code}
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {record.patient_name ?? '—'}
                        </p>
                      </div>
                      <Badge className={getBadgeClass(record.status)}>{record.status}</Badge>
                    </div>

                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between gap-4">
                        <span className="text-gray-600 dark:text-gray-400">Date:</span>
                        <span className="text-right text-gray-900 dark:text-white">
                          {formatDate(record.recording_date)}
                        </span>
                      </div>

                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Duration:</span>
                        <span className="text-gray-900 dark:text-white">{record.duration_label}</span>
                      </div>

                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Confidence:</span>
                        <span className="font-medium text-gray-900 dark:text-white">
                          {record.confidence != null ? `${record.confidence.toFixed(1)}%` : '—'}
                        </span>
                      </div>
                    </div>

                    <div className="flex gap-2 mt-4">
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex-1 dark:border-gray-700"
                        onClick={() => handleViewReport(record)}
                        disabled={!record.inference_id}
                      >
                        <Eye className="h-4 w-4 mr-2" />
                        View
                      </Button>

                      <Button
                        variant="outline"
                        size="sm"
                        className="flex-1 dark:border-gray-700"
                        onClick={() => handleDownloadReport(record)}
                        disabled={!record.inference_id}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Export
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          <Card className="dark:bg-slate-800 dark:border-gray-700">
            <CardContent className="p-6 flex items-center justify-between gap-4 flex-wrap">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Showing {filteredData.length === 0 ? 0 : (currentPage - 1) * itemsPerPage + 1} to{' '}
                {Math.min(currentPage * itemsPerPage, filteredData.length)} of {filteredData.length} records
              </p>

              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="dark:border-gray-700"
                  onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>

                <span className="text-sm text-gray-900 dark:text-white">
                  Page {currentPage} of {totalPages}
                </span>

                <Button
                  variant="outline"
                  size="sm"
                  className="dark:border-gray-700"
                  onClick={() => setCurrentPage((page) => Math.min(totalPages, page + 1))}
                  disabled={currentPage === totalPages}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}