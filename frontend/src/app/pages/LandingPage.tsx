import { Link } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { Moon, Sun, Activity, Brain, Shield, Zap, Upload, BarChart3, FileText } from 'lucide-react';
import { ImageWithFallback } from '../components/figma/ImageWithFallback';
import { Button } from '../components/ui/button';

export default function LandingPage() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="min-h-screen bg-white dark:bg-slate-900">
      {/* Navigation */}
      <nav className="border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16 sm:h-18">
            <div className="flex items-center gap-2">
              <Brain className="h-8 w-8 text-blue-600 dark:text-teal-400" />
              <span className="text-xl font-semibold text-gray-900 dark:text-white">
                NeuroXAI
              </span>
            </div>

            <div className="flex items-center gap-2 sm:gap-4">
              <button
                onClick={toggleTheme}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                {theme === 'light' ? (
                  <Moon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
                ) : (
                  <Sun className="h-5 w-5 text-gray-600 dark:text-gray-400" />
                )}
              </button>

              <Link to="/signin">
                <Button
                  variant="ghost"
                  className="text-gray-700 dark:text-gray-300 px-3 sm:px-4"
                >
                  Sign In
                </Button>
              </Link>

              <Link to="/signup">
                <Button className="bg-blue-600 hover:bg-blue-700 text-white dark:bg-teal-600 dark:hover:bg-teal-700 px-4 sm:px-6">
                  Get Started
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="py-12 sm:py-16 lg:py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-10 lg:gap-12 items-center">
            {/* Left Text Block */}
            <div className="order-1">
              <div className="max-w-2xl">
                <h1 className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-bold leading-tight text-gray-900 dark:text-white mb-6">
                  AI-Powered Neonatal
                  <br className="hidden sm:block" />
                  <span className="sm:hidden"> </span>
                  Seizure Detection
                </h1>

                <p className="text-base sm:text-lg lg:text-xl leading-7 sm:leading-8 text-gray-600 dark:text-gray-300 mb-8 max-w-xl">
                  NeuroXAI helps doctors detect seizures in newborns early using
                  advanced EEG analysis and explainable artificial intelligence.
                  Providing clinical decision support when it matters most.
                </p>

                <div className="flex flex-col sm:flex-row gap-4">
                  <Link to="/signup" className="w-full sm:w-auto">
                    <Button
                      size="lg"
                      className="w-full sm:w-auto bg-blue-600 hover:bg-blue-700 text-white dark:bg-teal-600 dark:hover:bg-teal-700 rounded-xl px-8"
                    >
                      Get Started
                    </Button>
                  </Link>

                  <Link to="/signin" className="w-full sm:w-auto">
                    <Button
                      size="lg"
                      variant="outline"
                      className="w-full sm:w-auto rounded-xl border-gray-300 text-gray-800 hover:bg-gray-50 dark:border-white/25 dark:bg-white/5 dark:text-white dark:hover:bg-white/10"
                    >
                      Sign In
                    </Button>
                  </Link>
                </div>
              </div>
            </div>

            {/* Right Image Block */}
            <div className="order-2 relative">
              <div className="relative overflow-hidden rounded-2xl sm:rounded-3xl shadow-2xl h-[320px] sm:h-[420px] lg:h-[560px] xl:h-[620px]">
                <ImageWithFallback
                  src="/images/neuroxai-baby.png"
                  alt="Neonatal EEG monitoring for seizure detection"
                  className="h-full w-full object-cover object-[72%_center]"
                />

                {/* light mode left fade */}
                <div className="hidden lg:block absolute inset-y-0 left-0 w-40 bg-gradient-to-r from-white via-white/70 to-transparent dark:hidden" />
                {/* dark mode left fade */}
                <div className="hidden lg:block absolute inset-y-0 left-0 w-40 bg-gradient-to-r from-slate-900 via-slate-900/75 to-transparent dark:block" />

                {/* soft image tint */}
                <div className="absolute inset-0 bg-black/5 dark:bg-slate-950/10" />
              </div>

              {/* glow behind image */}
              <div className="absolute -inset-4 -z-10 rounded-[2rem] bg-blue-200/30 blur-3xl dark:bg-cyan-500/10" />
            </div>
          </div>
        </div>
      </section>


      {/* How It Works */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gray-50 dark:bg-slate-800">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">How It Works</h2>
            <p className="text-xl text-gray-600 dark:text-gray-300">Simple, fast, and accurate seizure detection</p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white dark:bg-slate-900 p-8 rounded-xl shadow-lg">
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center mb-4">
                <Upload className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">Upload EEG Recording</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Upload EDF format EEG recordings from neonatal patients with secure, HIPAA-compliant processing.
              </p>
            </div>
            <div className="bg-white dark:bg-slate-900 p-8 rounded-xl shadow-lg">
              <div className="w-12 h-12 bg-teal-100 dark:bg-teal-900 rounded-lg flex items-center justify-center mb-4">
                <Activity className="h-6 w-6 text-teal-600 dark:text-teal-400" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">AI Analysis</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Our advanced AI model analyzes brain signals to detect seizure patterns with high accuracy.
              </p>
            </div>
            <div className="bg-white dark:bg-slate-900 p-8 rounded-xl shadow-lg">
              <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900 rounded-lg flex items-center justify-center mb-4">
                <BarChart3 className="h-6 w-6 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">Explainable Insights</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Get detailed explanations of AI predictions with visualizations showing which signals influenced the diagnosis.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Benefits */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Benefits</h2>
            <p className="text-xl text-gray-600 dark:text-gray-300">Transforming neonatal care with AI</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="p-6 border border-gray-200 dark:border-gray-700 rounded-xl">
              <Zap className="h-8 w-8 text-green-600 dark:text-green-400 mb-3" />
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Early Seizure Detection</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Identify seizures quickly to enable faster intervention and better outcomes.
              </p>
            </div>
            <div className="p-6 border border-gray-200 dark:border-gray-700 rounded-xl">
              <Brain className="h-8 w-8 text-blue-600 dark:text-blue-400 mb-3" />
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">AI-Assisted Diagnosis</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Reduce diagnostic burden with intelligent analysis that augments clinical expertise.
              </p>
            </div>
            <div className="p-6 border border-gray-200 dark:border-gray-700 rounded-xl">
              <FileText className="h-8 w-8 text-purple-600 dark:text-purple-400 mb-3" />
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Explainable Analysis</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Understand AI decisions with transparent visualizations of signal patterns.
              </p>
            </div>
            <div className="p-6 border border-gray-200 dark:border-gray-700 rounded-xl">
              <Shield className="h-8 w-8 text-teal-600 dark:text-teal-400 mb-3" />
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Clinical Decision Support</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Make informed decisions with confidence scores and detailed analysis reports.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200 dark:border-gray-800 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Brain className="h-6 w-6 text-blue-600 dark:text-teal-400" />
            <span className="text-lg font-semibold text-gray-900 dark:text-white">NeuroXAI</span>
          </div>
          <p className="text-gray-600 dark:text-gray-400">
            AI-Powered Neonatal Seizure Detection Platform
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-4">
            © 2026 NeuroXAI. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
