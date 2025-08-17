import React, { useRef, useEffect, useState } from 'react';
import axios from 'axios';

const VIDEO_SIZE = 224;

export default function CameraLive() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [streaming, setStreaming] = useState(false);

  useEffect(() => {
    // Start camera
    const video = videoRef.current;
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if (video) {
          video.srcObject = stream;
          setStreaming(true);
        }
      })
      .catch(() => setError('Could not access camera.'));
    return () => {
      if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (!streaming) return;
    const interval = setInterval(() => {
      sendFrame();
    }, 1000);
    return () => clearInterval(interval);
  }, [streaming]);

  const sendFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, VIDEO_SIZE, VIDEO_SIZE);
    canvasRef.current.toBlob(async (blob) => {
      if (!blob) return;
      setLoading(true);
      setError(null);
      const formData = new FormData();
      formData.append('image', blob, 'frame.jpg');
      try {
        const res = await axios.post('http://localhost:5000/predict', formData);
        setPrediction(res.data);
      } catch {
        setError('Prediction failed.');
      } finally {
        setLoading(false);
      }
    }, 'image/jpeg');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem', marginTop: '2rem' }}>
      <div style={{ position: 'relative', width: VIDEO_SIZE, height: VIDEO_SIZE }}>
        <video
          ref={videoRef}
          width={VIDEO_SIZE}
          height={VIDEO_SIZE}
          autoPlay
          muted
          style={{ borderRadius: 18, boxShadow: '0 6px 32px 0 rgba(0,0,0,0.18)', border: '3px solid var(--orange-yellow-crayola)', background: '#23272f' }}
        />
        {prediction && (
          <div style={{
            position: 'absolute',
            bottom: 10,
            left: 0,
            width: '100%',
            background: 'rgba(36,36,36,0.85)',
            color: 'var(--orange-yellow-crayola)',
            fontWeight: 600,
            fontSize: '1.1rem',
            borderRadius: '0 0 16px 16px',
            padding: '0.5rem 0',
            textAlign: 'center',
            letterSpacing: 0.5,
            boxShadow: '0 2px 8px rgba(0,0,0,0.12)'
          }}>
            {prediction.species ? `Species: ${prediction.species}` : ''}
            {prediction.confidence !== undefined ? ` | Confidence: ${(prediction.confidence * 100).toFixed(1)}%` : ''}
          </div>
        )}
        {loading && (
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            background: 'rgba(0,0,0,0.25)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ffd166',
            fontWeight: 600,
            fontSize: '1.2rem',
            borderRadius: 18
          }}>
            Predicting...
          </div>
        )}
      </div>
      <canvas ref={canvasRef} width={VIDEO_SIZE} height={VIDEO_SIZE} style={{ display: 'none' }} />
      {error && <div style={{ color: '#ff6f61', fontWeight: 500 }}>{error}</div>}
      <div style={{ color: 'var(--light-gray)', fontSize: '0.98rem', marginTop: '0.5rem' }}>
        {streaming ? 'Camera is live. Predictions update every second.' : 'Starting camera...'}
      </div>
    </div>
  );
} 