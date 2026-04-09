"use client";

import React, { useState } from "react";
import { GlassCard } from "@/components/ui/GlassCard";
import { GlassButton } from "@/components/ui/GlassButton";
import { FileText, CheckCircle, AlertTriangle, FileOutput } from "lucide-react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { ContextualChat } from "@/components/ContextualChat";

export default function ReportPage() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [activeTab, setActiveTab] = useState<
    "positive" | "negative" | "summary"
  >("summary");

  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      setProgress(10);
      setProgressMessage("Uploading report...");

      setTimeout(() => {
        setProgress(30);
        setProgressMessage("Sanitizing personal information...");
      }, 500);

      setTimeout(() => {
        setProgress(50);
        setProgressMessage("Analyzing medical data...");
      }, 1000);

      const response = await axios.post(
        "https://healthai-b6y2.onrender.com/report/analyze",
        formData,
      );

      setProgress(90);
      setProgressMessage("Finalizing results...");

      setTimeout(() => {
        setResults(response.data.results);
        setProgress(100);
        setProgressMessage("Analysis complete!");
        setIsLoading(false);
      }, 300);
    } catch (err: any) {
      console.error("Error analyzing report:", err);
      const errorMessage =
        err.response?.data?.detail ||
        "Failed to analyze report. Please try again.";
      setError(errorMessage);
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-white mb-2">Report Analyzer</h1>
        <p className="text-gray-400">
          Upload your medical report for instant AI analysis
        </p>
      </header>

      {!results ? (
        <GlassCard className="max-w-xl mx-auto text-center py-12">
          <div className="mb-8">
            <FileText className="w-20 h-20 mx-auto text-blue-400 mb-4 opacity-80" />
            <h3 className="text-xl font-semibold text-white mb-2">
              Upload Report
            </h3>
            <p className="text-gray-400">Supported formats: PDF, TXT</p>
          </div>

          <div className="flex flex-col items-center gap-4 max-w-xs mx-auto">
            <label className="w-full">
              <input
                type="file"
                accept=".pdf,.txt"
                className="hidden"
                onChange={handleFileChange}
              />
              <div className="w-full py-3 px-4 bg-glass-200 hover:bg-glass-300 rounded-xl cursor-pointer transition-colors text-white border border-glass-border flex items-center justify-center gap-2">
                <FileOutput size={18} />
                {file ? file.name : "Select File"}
              </div>
            </label>
            <GlassButton
              onClick={handleAnalyze}
              disabled={!file || isLoading}
              className="w-full"
            >
              {isLoading ? "Analyzing..." : "Analyze Report"}
            </GlassButton>
            {isLoading && (
              <ProgressBar
                progress={progress}
                message={progressMessage}
                className="mt-4"
              />
            )}
            {error && (
              <div className="mt-4 p-3 bg-red-500/20 border border-red-500/50 rounded-xl text-red-200 text-sm flex items-center gap-2">
                <AlertTriangle size={16} />
                {error}
              </div>
            )}
          </div>
        </GlassCard>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-1 grid grid-cols-3 lg:grid-cols-1 gap-2 lg:gap-2">
            <button
              onClick={() => setActiveTab("summary")}
              className={`w-full p-2 md:p-3 rounded-xl text-sm md:text-base text-center lg:text-left transition-all ${activeTab === "summary" ? "bg-blue-600/20 border border-blue-500 text-white" : "text-gray-400 hover:bg-glass-100"}`}
            >
              Summary
            </button>
            <button
              onClick={() => setActiveTab("positive")}
              className={`w-full p-2 md:p-3 rounded-xl text-sm md:text-base text-center lg:text-left transition-all ${activeTab === "positive" ? "bg-green-600/20 border border-green-500 text-white" : "text-gray-400 hover:bg-glass-100"}`}
            >
              Positive
            </button>
            <button
              onClick={() => setActiveTab("negative")}
              className={`w-full p-2 md:p-3 rounded-xl text-sm md:text-base text-center lg:text-left transition-all ${activeTab === "negative" ? "bg-red-600/20 border border-red-500 text-white" : "text-gray-400 hover:bg-glass-100"}`}
            >
              Concerns
            </button>
          </div>

          <GlassCard className="lg:col-span-3 min-h-[500px]">
            {activeTab === "summary" && results.summary_agent && (
              <div>
                <h3 className="text-xl font-semibold mb-4 text-white flex items-center gap-2">
                  <FileText className="text-blue-400" /> Executive Summary
                </h3>
                <div className="prose prose-invert max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {results.summary_agent.content}
                  </ReactMarkdown>
                </div>
              </div>
            )}

            {activeTab === "positive" && results.positive_analyzer && (
              <div>
                <h3 className="text-xl font-semibold mb-4 text-white flex items-center gap-2">
                  <CheckCircle className="text-green-400" /> Positive Findings
                </h3>
                <div className="prose prose-invert max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {results.positive_analyzer.content}
                  </ReactMarkdown>
                </div>
              </div>
            )}

            {activeTab === "negative" && results.negative_analyzer && (
              <div>
                <h3 className="text-xl font-semibold mb-4 text-white flex items-center gap-2">
                  <AlertTriangle className="text-red-400" /> Areas of Concern
                </h3>
                <div className="prose prose-invert max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {results.negative_analyzer.content}
                  </ReactMarkdown>
                </div>
              </div>
            )}
          </GlassCard>
        </div>
      )}

      {results && (
        <ContextualChat 
          featureType="Medical Report" 
          originalResult={JSON.stringify(results, null, 2)} 
        />
      )}
    </div>
  );
}
