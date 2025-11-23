import Link from "next/link";
import { Activity, MessageSquare, Utensils, FileText, Image as ImageIcon, Scan } from "lucide-react";

export default function Home() {
    const features = [
        {
            title: "Health Assistant",
            description: "Chat with AI about your health concerns",
            icon: <MessageSquare className="w-8 h-8 mb-4 text-blue-400" />,
            href: "/chat",
            color: "from-blue-500/20 to-cyan-500/20"
        },
        {
            title: "Diet Planner",
            description: "Get personalized meal plans & food analysis",
            icon: <Utensils className="w-8 h-8 mb-4 text-green-400" />,
            href: "/diet",
            color: "from-green-500/20 to-emerald-500/20"
        },
        {
            title: "Health Prediction",
            description: "Assess risks for diabetes, heart health & more",
            icon: <Activity className="w-8 h-8 mb-4 text-red-400" />,
            href: "/prediction",
            color: "from-red-500/20 to-pink-500/20"
        },
        {
            title: "Report Analysis",
            description: "Understand your medical reports instantly",
            icon: <FileText className="w-8 h-8 mb-4 text-yellow-400" />,
            href: "/report",
            color: "from-yellow-500/20 to-orange-500/20"
        },
        {
            title: "Image Analysis",
            description: "AI analysis of medical images",
            icon: <ImageIcon className="w-8 h-8 mb-4 text-purple-400" />,
            href: "/image-analysis",
            color: "from-purple-500/20 to-violet-500/20"
        },
        {
            title: "Anomaly Detection",
            description: "Advanced AI-powered anomaly detection",
            icon: <Scan className="w-8 h-8 mb-4 text-teal-400" />,
            href: "/anomaly-detection",
            color: "from-teal-500/20 to-cyan-500/20",
            comingSoon: true
        }
    ];

    return (
        <div className="max-w-7xl mx-auto">
            <header className="mb-12 md:mb-20 text-center px-4">
                <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 mb-4 md:mb-6 tracking-tight">
                    HealthAI
                </h1>
                <p className="text-lg md:text-xl text-gray-300 max-w-2xl mx-auto leading-relaxed">
                    Your intelligent companion for a healthier life. Experience the future of personal healthcare management.
                </p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {features.map((feature, index) => (
                    <Link
                        key={index}
                        href={feature.href}
                        className={`glass-panel p-8 hover:scale-[1.02] transition-all duration-300 group relative overflow-hidden`}
                    >
                        <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
                        <div className="relative z-10">
                            {feature.icon}
                            <div className="flex items-center gap-2 mb-2">
                                <h2 className="text-2xl font-semibold text-white">{feature.title}</h2>
                                {feature.comingSoon && (
                                    <span className="px-2 py-1 text-xs font-semibold bg-teal-500/20 text-teal-300 rounded-full border border-teal-500/30">
                                        Coming Soon
                                    </span>
                                )}
                            </div>
                            <p className="text-gray-400 group-hover:text-gray-200 transition-colors">
                                {feature.description}
                            </p>
                        </div>
                    </Link>
                ))}
            </div>
        </div>
    );
}
