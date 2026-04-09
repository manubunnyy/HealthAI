import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/Navbar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "HealthAI - Liquid Glass",
  description: "Advanced Health AI Assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="fixed inset-0 -z-10">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse delay-1000" />
        </div>
        <Navbar />
        <main className="min-h-screen px-4 py-6 md:px-8 md:py-12 pt-24 md:pt-28 max-w-[1920px] mx-auto">
          {children}
        </main>
      </body>
    </html>
  );
}
