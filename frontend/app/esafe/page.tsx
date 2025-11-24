"use client";

import { useState, useRef, useEffect } from 'react';
import { Ambulance, Car, Heart, Baby, MapPin, Upload, Send, AlertTriangle, ArrowLeft, Settings, X } from 'lucide-react';
import dynamic from 'next/dynamic';

// Dynamically import Map component to avoid SSR issues
const LeafletMap = dynamic(() => import('../../components/LeafletMap'), {
    ssr: false,
    loading: () => <div className="h-[300px] w-full bg-glass-100 animate-pulse rounded-xl flex items-center justify-center text-gray-500">Loading Map...</div>
});

export default function ESafePage() {
    const [step, setStep] = useState<'type' | 'location' | 'details' | 'success'>('type');
    const [emergencyType, setEmergencyType] = useState<string>('');
    const [location, setLocation] = useState<{ lat: number; lng: number } | null>(null);
    const [address, setAddress] = useState<string>('');
    const [photos, setPhotos] = useState<File[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Admin Settings State
    const [showSettings, setShowSettings] = useState(false);
    const [emailConfig, setEmailConfig] = useState({
        sender_email: '',
        sender_password: '',
        receiver_email: ''
    });
    const [savingConfig, setSavingConfig] = useState(false);

    const emergencyTypes = [
        { id: 'Medical Emergency', icon: Ambulance, color: 'text-red-500', border: 'border-red-500/50', gradient: 'from-red-500/20 to-pink-500/20' },
        { id: 'Accident', icon: Car, color: 'text-orange-500', border: 'border-orange-500/50', gradient: 'from-orange-500/20 to-yellow-500/20' },
        { id: 'Heart/Chest Pain', icon: Heart, color: 'text-pink-500', border: 'border-pink-500/50', gradient: 'from-pink-500/20 to-rose-500/20' },
        { id: 'Pregnancy', icon: Baby, color: 'text-purple-500', border: 'border-purple-500/50', gradient: 'from-purple-500/20 to-violet-500/20' },
    ];

    useEffect(() => {
        if (showSettings) {
            fetch('http://localhost:8000/esafe/config')
                .then(res => res.json())
                .then(data => {
                    if (data.sender_email) {
                        setEmailConfig(prev => ({ ...prev, ...data, sender_password: '' }));
                    }
                })
                .catch(err => console.error("Failed to load config", err));
        }
    }, [showSettings]);

    const handleLocation = () => {
        setLoading(true);
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    setLocation({
                        lat: position.coords.latitude,
                        lng: position.coords.longitude
                    });
                    setLoading(false);
                    // Don't auto-advance, let user see map
                },
                (err) => {
                    setError("Could not get location. Please enter address manually.");
                    setLoading(false);
                }
            );
        } else {
            setError("Geolocation is not supported by this browser.");
            setLoading(false);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setPhotos(Array.from(e.target.files));
        }
    };

    const handleSubmit = async () => {
        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('type', emergencyType);
            if (location) {
                formData.append('latitude', location.lat.toString());
                formData.append('longitude', location.lng.toString());
            }
            if (address) {
                formData.append('text_address', address);
            }
            photos.forEach(photo => {
                formData.append('photos', photo);
            });

            const response = await fetch('http://localhost:8000/esafe/alert', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Failed to send alert');
            }

            setStep('success');
        } catch (err) {
            setError('Failed to send emergency alert. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleSaveConfig = async () => {
        setSavingConfig(true);
        try {
            const response = await fetch('http://localhost:8000/esafe/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(emailConfig)
            });
            if (response.ok) {
                setShowSettings(false);
                alert("Settings saved successfully!");
            } else {
                alert("Failed to save settings");
            }
        } catch (e) {
            alert("Error saving settings");
        } finally {
            setSavingConfig(false);
        }
    };

    const handleBack = () => {
        if (step === 'location') setStep('type');
        if (step === 'details') setStep('location');
    };

    return (
        <div className="min-h-screen text-white p-4 pb-24 pt-24">
            <div className="max-w-2xl mx-auto space-y-8 relative">
                {/* Header Controls */}
                <div className="flex justify-between items-center absolute w-full -top-12 md:top-2 px-2">
                    {step !== 'type' && step !== 'success' ? (
                        <button
                            onClick={handleBack}
                            className="p-2 text-gray-400 hover:text-white transition-colors flex items-center gap-2 hover:bg-white/10 rounded-lg"
                        >
                            <ArrowLeft size={20} />
                            <span>Back</span>
                        </button>
                    ) : <div></div>}

                    <button
                        onClick={() => setShowSettings(true)}
                        className="p-2 text-gray-400 hover:text-white transition-colors hover:bg-white/10 rounded-lg"
                    >
                        <Settings size={20} />
                    </button>
                </div>

                <div className="text-center space-y-2 pt-8 md:pt-0">
                    <h1 className="text-4xl font-bold text-red-500 flex items-center justify-center gap-3">
                        <AlertTriangle size={40} />
                        Emergency Assistance
                    </h1>
                    <p className="text-gray-300">Get help quickly in emergency situations</p>
                </div>

                {error && (
                    <div className="glass-panel border-red-500/50 p-4 text-red-200 text-center bg-red-900/20">
                        {error}
                    </div>
                )}

                {step === 'type' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {emergencyTypes.map((type) => (
                            <button
                                key={type.id}
                                onClick={() => {
                                    setEmergencyType(type.id);
                                    setStep('location');
                                }}
                                className={`glass-panel p-6 border ${type.border} transition-all hover:scale-105 group relative overflow-hidden`}
                            >
                                <div className={`absolute inset-0 bg-gradient-to-br ${type.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
                                <div className="relative z-10 flex flex-col items-center gap-4">
                                    <type.icon size={48} className={`${type.color} group-hover:scale-110 transition-transform`} />
                                    <span className="text-xl font-semibold">{type.id}</span>
                                </div>
                            </button>
                        ))}
                    </div>
                )}

                {step === 'location' && (
                    <div className="space-y-6">
                        <div className="glass-panel p-8 space-y-6">
                            <h2 className="text-2xl font-semibold text-center">Share Location</h2>

                            {location && (
                                <div className="animate-in fade-in zoom-in duration-300">
                                    <LeafletMap lat={location.lat} lng={location.lng} />
                                </div>
                            )}

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <button
                                    onClick={handleLocation}
                                    disabled={loading}
                                    className="glass-button bg-blue-600/80 hover:bg-blue-600 flex items-center justify-center gap-2 disabled:opacity-50"
                                >
                                    {loading ? (
                                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                                    ) : (
                                        <>
                                            <MapPin /> {location ? "Update Location" : "Share Current Location"}
                                        </>
                                    )}
                                </button>
                                <button
                                    onClick={() => setStep('details')}
                                    className={`glass-button ${location ? 'bg-green-600/80 hover:bg-green-600' : 'bg-glass-200 hover:bg-glass-300'} flex items-center justify-center gap-2`}
                                >
                                    {location ? "Continue with Location" : "Enter Address Manually"}
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {step === 'details' && (
                    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4">
                        <div className="glass-panel p-8 space-y-6">
                            <h2 className="text-2xl font-semibold">Additional Details</h2>

                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-gray-300">Address / Landmark</label>
                                <textarea
                                    value={address}
                                    onChange={(e) => setAddress(e.target.value)}
                                    placeholder="Enter complete address or nearby landmarks..."
                                    className="glass-input w-full min-h-[100px]"
                                />
                            </div>

                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-gray-300">Photos (Optional)</label>
                                <div
                                    onClick={() => fileInputRef.current?.click()}
                                    className="glass-input border-dashed border-2 hover:border-gray-400 p-8 text-center cursor-pointer transition-colors flex flex-col items-center justify-center"
                                >
                                    <input
                                        type="file"
                                        ref={fileInputRef}
                                        onChange={handleFileChange}
                                        multiple
                                        accept="image/*"
                                        className="hidden"
                                    />
                                    <Upload className="h-12 w-12 text-gray-400 mb-4" />
                                    <p className="text-gray-400">Click to upload photos of the situation</p>
                                    {photos.length > 0 && (
                                        <p className="text-green-400 mt-2 font-medium">{photos.length} photo(s) selected</p>
                                    )}
                                </div>
                            </div>

                            <button
                                onClick={handleSubmit}
                                disabled={loading}
                                className="glass-button w-full bg-red-600/80 hover:bg-red-600 font-bold text-lg flex items-center justify-center gap-2 shadow-lg shadow-red-900/20"
                            >
                                {loading ? (
                                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                                ) : (
                                    <>
                                        <Send /> Send Emergency Alert
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                )}

                {step === 'success' && (
                    <div className="text-center space-y-6 animate-in zoom-in duration-300">
                        <div className="w-24 h-24 bg-green-500 rounded-full flex items-center justify-center mx-auto shadow-xl shadow-green-500/20">
                            <Send size={48} className="text-white" />
                        </div>
                        <h2 className="text-3xl font-bold text-green-500">Alert Sent Successfully!</h2>
                        <div className="glass-panel p-8 border-green-500/30 space-y-4">
                            <p className="text-xl">Help is on the way!</p>
                            <p className="text-gray-400">Estimated arrival time: <span className="text-white font-bold">5-15 minutes</span></p>

                            <div className="border-t border-glass-border pt-4 mt-4 text-left space-y-2">
                                <p className="font-semibold text-yellow-500">Important Instructions:</p>
                                <ul className="list-disc list-inside text-gray-300 space-y-1">
                                    <li>Stay calm and remain in your current location</li>
                                    <li>Keep your phone nearby</li>
                                    <li>Clear the path for emergency responders</li>
                                </ul>
                            </div>
                        </div>

                        <button
                            onClick={() => {
                                setStep('type');
                                setEmergencyType('');
                                setLocation(null);
                                setAddress('');
                                setPhotos([]);
                            }}
                            className="text-gray-400 hover:text-white underline transition-colors"
                        >
                            Start New Emergency Request
                        </button>
                    </div>
                )}

                {/* Settings Modal */}
                {showSettings && (
                    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
                        <div className="glass-panel p-8 max-w-md w-full space-y-6 animate-in zoom-in duration-200">
                            <div className="flex justify-between items-center">
                                <h2 className="text-2xl font-bold">Admin Settings</h2>
                                <button onClick={() => setShowSettings(false)} className="text-gray-400 hover:text-white">
                                    <X size={24} />
                                </button>
                            </div>

                            <div className="space-y-4">
                                <div className="space-y-2">
                                    <label className="text-sm text-gray-300">Sender Email (Gmail)</label>
                                    <input
                                        type="email"
                                        value={emailConfig.sender_email}
                                        onChange={e => setEmailConfig({ ...emailConfig, sender_email: e.target.value })}
                                        className="glass-input w-full"
                                        placeholder="your-email@gmail.com"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm text-gray-300">App Password</label>
                                    <input
                                        type="password"
                                        value={emailConfig.sender_password}
                                        onChange={e => setEmailConfig({ ...emailConfig, sender_password: e.target.value })}
                                        className="glass-input w-full"
                                        placeholder="App Password (not login password)"
                                    />
                                    <p className="text-xs text-gray-500">Use an App Password from Google Account settings.</p>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm text-gray-300">Receiver Email</label>
                                    <input
                                        type="email"
                                        value={emailConfig.receiver_email}
                                        onChange={e => setEmailConfig({ ...emailConfig, receiver_email: e.target.value })}
                                        className="glass-input w-full"
                                        placeholder="admin@example.com"
                                    />
                                </div>
                            </div>

                            <button
                                onClick={handleSaveConfig}
                                disabled={savingConfig}
                                className="glass-button w-full bg-blue-600/80 hover:bg-blue-600 font-bold"
                            >
                                {savingConfig ? "Saving..." : "Save Configuration"}
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
