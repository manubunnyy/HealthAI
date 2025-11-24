import Link from 'next/link';
import { Home, Ambulance } from 'lucide-react';

export function Navbar() {
    return (
        <nav className="fixed top-4 left-4 z-50 flex gap-4">
            <Link
                href="/"
                className="flex items-center gap-2 bg-glass-200 hover:bg-glass-300 backdrop-blur-md border border-glass-border rounded-full px-5 py-2.5 text-white transition-all shadow-lg hover:scale-105 active:scale-95"
            >
                <Home size={20} />
                <span className="font-medium">Home</span>
            </Link>
            <Link
                href="/esafe"
                className="flex items-center gap-2 bg-red-600/20 hover:bg-red-600/30 backdrop-blur-md border border-red-500/30 rounded-full px-5 py-2.5 text-red-100 transition-all shadow-lg hover:scale-105 active:scale-95"
            >
                <Ambulance size={20} />
                <span className="font-medium">eSafe</span>
            </Link>
        </nav>
    );
}
