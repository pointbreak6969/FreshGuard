'use client';

import { useState, useRef } from 'react';

interface Detection {
  label: string;
  confidence: number;
  path: string;
}

export default function Home() {
  const [image, setImage] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImage(e.target?.result as string);
        setDetections([]);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDetect = async () => {
    if (!fileInputRef.current?.files?.[0]) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append('file', fileInputRef.current.files[0]);

    try {
      const response = await fetch('http://localhost:5000/detect', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setDetections(data.detections || []);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-green-100 font-sans">
      <main className="container mx-auto px-4 py-12 max-w-2xl">
        <h1 className="text-4xl font-bold text-center text-green-800 mb-2">
          FreshGuard
        </h1>
        <p className="text-center text-green-600 mb-8">
          Detect fruits & vegetables from your photo
        </p>

        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="border-3 border-dashed border-green-300 rounded-xl p-8 text-center mb-6">
            {image ? (
              <div className="relative inline-block">
                <img 
                  src={image} 
                  alt="Selected" 
                  className="max-h-64 rounded-lg"
                />
                <button
                  onClick={() => { setImage(null); setDetections([]); if (fileInputRef.current) fileInputRef.current.value = ''; }}
                  className="absolute top-2 right-2 bg-red-500 text-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-red-600"
                >
                  ×
                </button>
              </div>
            ) : (
              <div className="py-12">
                <svg className="mx-auto h-16 w-16 text-green-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p className="text-gray-600 mb-4">Select a photo to detect fruits & vegetables</p>
              </div>
            )}
            
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
              id="file-input"
            />
            
            <label
              htmlFor="file-input"
              className="inline-block bg-green-600 text-white px-6 py-3 rounded-full font-medium cursor-pointer hover:bg-green-700 transition-colors"
            >
              {image ? 'Choose Different Photo' : 'Choose Photo'}
            </label>
          </div>

          {image && (
            <button
              onClick={handleDetect}
              disabled={loading}
              className="w-full bg-green-600 text-white py-3 rounded-xl font-semibold text-lg hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {loading ? 'Detecting...' : 'Detect Produce'}
            </button>
          )}

          {detections.length > 0 && (
            <div className="mt-8">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Detected Items:</h2>
              <div className="space-y-3">
                {detections.map((det, idx) => (
                  <div key={idx} className="flex items-center justify-between bg-green-50 rounded-lg p-4">
                    <span className="font-medium text-gray-800 capitalize">{det.label}</span>
                    <span className="text-green-700 font-semibold">{Math.round(det.confidence * 100)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {detections.length === 0 && !loading && image && (
            <div className="mt-8 text-center text-gray-500">
              No fruits or vegetables detected in this image.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}