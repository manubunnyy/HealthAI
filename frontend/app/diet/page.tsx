"use client";

import React, { useState } from "react";
import { GlassCard } from "@/components/ui/GlassCard";
import { GlassButton } from "@/components/ui/GlassButton";
import { GlassInput } from "@/components/ui/GlassInput";
import { Upload, Utensils, Sparkles } from "lucide-react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ContextualChat } from "@/components/ContextualChat";

export default function DietPage() {
  const [activeTab, setActiveTab] = useState<"scan" | "plan">("scan");
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const [planInput, setPlanInput] = useState("");
  const [generatedPlan, setGeneratedPlan] = useState<string | null>(null);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnalysis(null);
    }
  };

  const handleAnalyzeFood = async () => {
    if (!selectedImage) return;
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const response = await axios.post(
        "https://healthai-b6y2.onrender.com/diet/analyze-food",
        formData,
      );
      setAnalysis(response.data.analysis);
    } catch (error) {
      console.error("Error analyzing food:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGeneratePlan = async () => {
    if (!planInput.trim()) return;
    setIsLoading(true);

    try {
      const response = await axios.post(
        "https://healthai-b6y2.onrender.com/diet/generate-plan",
        { user_input: planInput },
      );
      setGeneratedPlan(response.data.plan);
    } catch (error) {
      console.error("Error generating plan:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-white mb-2">Smart Dietitian</h1>
        <p className="text-gray-400">
          AI-powered food analysis and meal planning
        </p>
      </header>

      <div className="flex justify-center mb-8">
        <div className="bg-glass-100 p-1 rounded-xl inline-flex">
          <button
            onClick={() => setActiveTab("scan")}
            className={`px-6 py-2 rounded-lg transition-all ${activeTab === "scan" ? "bg-blue-600 text-white shadow-lg" : "text-gray-400 hover:text-white"}`}
          >
            📸 Food Scanner
          </button>
          <button
            onClick={() => setActiveTab("plan")}
            className={`px-6 py-2 rounded-lg transition-all ${activeTab === "plan" ? "bg-blue-600 text-white shadow-lg" : "text-gray-400 hover:text-white"}`}
          >
            🥗 Meal Planner
          </button>
        </div>
      </div>

      {activeTab === "scan" ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <GlassCard className="flex flex-col items-center justify-center min-h-[400px]">
            {previewUrl ? (
              <div className="relative w-full h-full min-h-[300px] rounded-xl overflow-hidden mb-4">
                <img
                  src={previewUrl}
                  alt="Food preview"
                  className="w-full h-full object-cover"
                />
              </div>
            ) : (
              <div className="text-center p-8 border-2 border-dashed border-glass-border rounded-xl w-full h-full flex flex-col items-center justify-center text-gray-400">
                <Utensils className="w-16 h-16 mb-4 opacity-50" />
                <p>Upload a photo of your meal</p>
              </div>
            )}

            <div className="flex gap-4 w-full mt-4">
              <label className="flex-1">
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleImageSelect}
                />
                <div className="w-full py-3 bg-glass-200 hover:bg-glass-300 rounded-xl text-center cursor-pointer transition-colors text-white">
                  Select Image
                </div>
              </label>
              <GlassButton
                onClick={handleAnalyzeFood}
                disabled={!selectedImage || isLoading}
                className="flex-1"
              >
                {isLoading ? "Analyzing..." : "Analyze Nutrition"}
              </GlassButton>
            </div>
          </GlassCard>

          <GlassCard className="min-h-[400px]">
            <h3 className="text-xl font-semibold mb-4 text-white flex items-center gap-2">
              <Sparkles className="text-yellow-400" size={20} />
              Analysis Results
            </h3>
            {analysis ? (
              <div className="prose prose-invert max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {analysis}
                </ReactMarkdown>
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500">
                Analysis results will appear here
              </div>
            )}
          </GlassCard>
        </div>
      ) : (
        <GlassCard>
          <h3 className="text-xl font-semibold mb-4 text-white">
            Personalized Meal Plan
          </h3>
          <p className="text-gray-400 mb-4">
            Tell us about your available ingredients, dietary preferences, and
            goals.
          </p>

          <textarea
            className="w-full bg-glass-100 backdrop-blur-sm border border-glass-border rounded-xl p-4 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 min-h-[150px] mb-4 placeholder-gray-500"
            placeholder="Example: I have chicken, rice, and broccoli. I want to build muscle and I'm lactose intolerant."
            value={planInput}
            onChange={(e) => setPlanInput(e.target.value)}
          />

          <GlassButton
            onClick={handleGeneratePlan}
            disabled={isLoading || !planInput.trim()}
            className="w-full mb-8"
          >
            {isLoading ? "Generating Plan..." : "Generate Meal Plan"}
          </GlassButton>

          {generatedPlan && (
            <div className="mt-8 pt-8 border-t border-glass-border">
              <h4 className="text-lg font-semibold mb-4 text-green-400">
                Your Custom Plan
              </h4>
              <div className="prose prose-invert max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {generatedPlan}
                </ReactMarkdown>
              </div>
            </div>
          )}
        </GlassCard>
      )}

      {/* Render Contextual Chat if a result exists */}
      {activeTab === "scan" && analysis && (
        <ContextualChat featureType="Food Analysis" originalResult={analysis} />
      )}
      {activeTab === "plan" && generatedPlan && (
        <ContextualChat featureType="Diet Plan" originalResult={generatedPlan} />
      )}
    </div>
  );
}
