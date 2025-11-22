import Link from 'next/link';
import { Home } from 'lucide-react';

export function Navbar() {
    return (
        <nav className="fixed top-4 left-4 z-50">
            <Link
                href="/"
                className="flex items-center gap-2 bg-glass-200 hover:bg-glass-300 backdrop-blur-md border border-glass-border rounded-full px-5 py-2.5 text-white transition-all shadow-lg hover:scale-105 active:scale-95"
            >
                <Home size={20} />
                <span className="font-medium">Home</span>
            </Link>
        </nav>
    );
}
