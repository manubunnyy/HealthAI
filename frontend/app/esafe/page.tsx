"use client";

import { useState, useRef } from 'react';
import { Ambulance, Car, Heart, Baby, MapPin, Upload, Send, AlertTriangle } from 'lucide-react';

export default function ESafePage() {
    const [step, setStep] = useState<'type' | 'location' | 'details' | 'success'>('type');
    const [emergencyType, setEmergencyType] = useState<string>('');
    const [location, setLocation] = useState<{ lat: number; lng: number } | null>(null);
    const [address, setAddress] = useState<string>('');
    const [photos, setPhotos] = useState<File[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const emergencyTypes = [
        { id: 'Medical Emergency', icon: Ambulance, color: 'text-red-500', border: 'border-red-500/50' },
        { id: 'Accident', icon: Car, color: 'text-orange-500', border: 'border-orange-500/50' },
        { id: 'Heart/Chest Pain', icon: Heart, color: 'text-pink-500', border: 'border-pink-500/50' },
        { id: 'Pregnancy', icon: Baby, color: 'text-purple-500', border: 'border-purple-500/50' },
    ];

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
                    setStep('details');
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

    return (
        <div className="min-h-screen bg-black text-white p-4 pb-24 pt-24">
            <div className="max-w-2xl mx-auto space-y-8">
                <div className="text-center space-y-2">
                    <h1 className="text-4xl font-bold text-red-500 flex items-center justify-center gap-3">
                        <AlertTriangle size={40} />
                        Emergency Assistance
                    </h1>
                    <p className="text-gray-400">Get help quickly in emergency situations</p>
                </div>

                {error && (
                    <div className="bg-red-900/20 border border-red-500/50 p-4 rounded-xl text-red-200 text-center">
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
                                className={`p-6 bg-glass-100 hover:bg-glass-200 border ${type.border} rounded-2xl transition-all hover:scale-105 flex flex-col items-center gap-4 group`}
                            >
                                <type.icon size={48} className={`${type.color} group-hover:scale-110 transition-transform`} />
                                <span className="text-xl font-semibold">{type.id}</span>
                            </button>
                        ))}
                    </div>
                )}

                {step === 'location' && (
                    <div className="space-y-6">
                        <div className="bg-glass-100 p-6 rounded-2xl border border-glass-border space-y-4">
                            <h2 className="text-2xl font-semibold text-center">Share Location</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <button
                                    onClick={handleLocation}
                                    disabled={loading}
                                    className="p-4 bg-blue-600 hover:bg-blue-700 rounded-xl flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                                >
                                    {loading ? (
                                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                                    ) : (
                                        <>
                                            <MapPin /> Share Current Location
                                        </>
                                    )}
                                </button>
                                <button
                                    onClick={() => setStep('details')}
                                    className="p-4 bg-gray-700 hover:bg-gray-600 rounded-xl flex items-center justify-center gap-2 transition-colors"
                                >
                                    Enter Address Manually
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {step === 'details' && (
                    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4">
                        <div className="bg-glass-100 p-6 rounded-2xl border border-glass-border space-y-6">
                            <h2 className="text-2xl font-semibold">Additional Details</h2>

                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-gray-300">Address / Landmark</label>
                                <textarea
                                    value={address}
                                    onChange={(e) => setAddress(e.target.value)}
                                    placeholder="Enter complete address or nearby landmarks..."
                                    className="w-full p-4 bg-black/50 border border-gray-700 rounded-xl focus:border-red-500 focus:ring-1 focus:ring-red-500 outline-none transition-all min-h-[100px]"
                                />
                            </div>

                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-gray-300">Photos (Optional)</label>
                                <div
                                    onClick={() => fileInputRef.current?.click()}
                                    className="border-2 border-dashed border-gray-700 hover:border-gray-500 rounded-xl p-8 text-center cursor-pointer transition-colors"
                                >
                                    <input
                                        type="file"
                                        ref={fileInputRef}
                                        onChange={handleFileChange}
                                        multiple
                                        accept="image/*"
                                        className="hidden"
                                    />
                                    <Upload className="mx-auto h-12 w-12 text-gray-500 mb-4" />
                                    <p className="text-gray-400">Click to upload photos of the situation</p>
                                    {photos.length > 0 && (
                                        <p className="text-green-500 mt-2">{photos.length} photo(s) selected</p>
                                    )}
                                </div>
                            </div>

                            <button
                                onClick={handleSubmit}
                                disabled={loading}
                                className="w-full p-4 bg-red-600 hover:bg-red-700 rounded-xl font-bold text-lg flex items-center justify-center gap-2 transition-all hover:scale-[1.02] disabled:opacity-50 disabled:hover:scale-100 shadow-lg shadow-red-900/20"
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
                        <div className="bg-glass-100 p-6 rounded-2xl border border-green-500/30 space-y-4">
                            <p className="text-xl">Help is on the way!</p>
                            <p className="text-gray-400">Estimated arrival time: <span className="text-white font-bold">5-15 minutes</span></p>

                            <div className="border-t border-gray-700 pt-4 mt-4 text-left space-y-2">
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
                            className="text-gray-400 hover:text-white underline"
                        >
                            Start New Emergency Request
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
