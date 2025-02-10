import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Box, Button, FormControl, InputLabel, Select, MenuItem } from '@mui/material';

const SimpleWebcamTest = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const frameRequestRef = useRef(null);
    const lastFrameTime = useRef(0);
    const [isStreaming, setIsStreaming] = useState(false);
    const [cameras, setCameras] = useState([]);
    const [selectedCamera, setSelectedCamera] = useState('');
    const [debugInfo, setDebugInfo] = useState({
        framesReceived: 0,
        lastFrameTime: null
    });

    // Get available cameras
    useEffect(() => {
        async function getCameras() {
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(device => device.kind === 'videoinput');
                setCameras(videoDevices);
                if (videoDevices.length > 0) {
                    setSelectedCamera(videoDevices[0].deviceId);
                }
            } catch (err) {
                console.error("Error getting cameras:", err);
            }
        }
        getCameras();
    }, []);

    const sendFrame = useCallback(() => {
        const FPS = 15; // Lower FPS for better performance
        const frameInterval = 1000 / FPS;
        const now = performance.now();
        
        if (!videoRef.current || !canvasRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            frameRequestRef.current = requestAnimationFrame(sendFrame);
            return;
        }
    
        if (now - lastFrameTime.current < frameInterval) {
            frameRequestRef.current = requestAnimationFrame(sendFrame);
            return;
        }
        
        lastFrameTime.current = now;
    
        const tempCanvas = document.createElement('canvas');
        // Reduce resolution for better performance
        tempCanvas.width = 640;  // Half resolution
        tempCanvas.height = 480; // Half resolution
        const tempContext = tempCanvas.getContext('2d');
        tempContext.drawImage(videoRef.current, 0, 0, tempCanvas.width, tempCanvas.height);
    
        tempCanvas.toBlob(
            (blob) => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    if (wsRef.current?.readyState === WebSocket.OPEN) {
                        wsRef.current.send(reader.result);
                        frameRequestRef.current = requestAnimationFrame(sendFrame);
                    }
                };
                reader.readAsDataURL(blob);
            },
            'image/jpeg',
            0.7  // Lower quality for better performance
        );
    }, []);

    const startStream = useCallback(async () => {
        try {
            // Stop any existing stream
            if (videoRef.current?.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            }
            if (frameRequestRef.current) {
                cancelAnimationFrame(frameRequestRef.current);
            }

            // Get webcam stream
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: selectedCamera ? { exact: selectedCamera } : undefined,
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 }
                }
            });

            // Set up video
            videoRef.current.srcObject = stream;
            await videoRef.current.play();

            // Set up canvas
            const videoTrack = stream.getVideoTracks()[0];
            const settings = videoTrack.getSettings();
            canvasRef.current.width = settings.width || 640;
            canvasRef.current.height = settings.height || 480;
            console.log("Canvas dimensions set to:", canvasRef.current.width, "x", canvasRef.current.height);

            // Set up WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${import.meta.env.VITE_API_URL.replace(/^https?:\/\//, '')}/ws`;
            console.log("Connecting to WebSocket:", wsUrl);
            
            wsRef.current = new WebSocket(wsUrl);

            wsRef.current.onopen = () => {
                console.log('WebSocket Connected');
                setIsStreaming(true);
                frameRequestRef.current = requestAnimationFrame(sendFrame);
            };

            wsRef.current.onmessage = (event) => {
                const data = JSON.parse(event.data);
                setDebugInfo(prev => ({
                    framesReceived: prev.framesReceived + 1,
                    lastFrameTime: new Date().toISOString()
                }));
                
                if (data.image) {
                    const img = new Image();
                    img.onload = () => {
                        const ctx = canvasRef.current.getContext('2d');
                        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                        ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height);
                    };
                    img.src = data.image;
                }
            };

            wsRef.current.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            wsRef.current.onclose = () => {
                console.log('WebSocket closed');
                if (isStreaming) {
                    stopStream();
                }
            };

        } catch (err) {
            console.error("Error:", err);
        }
    }, [selectedCamera, isStreaming, sendFrame]);

    const stopStream = useCallback(() => {
        setIsStreaming(false);
        if (frameRequestRef.current) {
            cancelAnimationFrame(frameRequestRef.current);
        }
        if (videoRef.current?.srcObject) {
            videoRef.current.srcObject.getTracks().forEach(track => track.stop());
        }
        if (wsRef.current) {
            wsRef.current.close();
        }
    }, []);

    useEffect(() => {
        return () => {
            if (frameRequestRef.current) {
                cancelAnimationFrame(frameRequestRef.current);
            }
        };
    }, []);

    const handleCameraChange = (event) => {
        setSelectedCamera(event.target.value);
        if (isStreaming) {
            stopStream();
            setTimeout(() => startStream(), 100);
        }
    };

    return (
        <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'center' }}>
                <FormControl sx={{ minWidth: 200 }}>
                    <InputLabel>Camera</InputLabel>
                    <Select
                        value={selectedCamera}
                        onChange={handleCameraChange}
                        label="Camera"
                    >
                        {cameras.map((camera) => (
                            <MenuItem key={camera.deviceId} value={camera.deviceId}>
                                {camera.label || `Camera ${camera.deviceId.slice(0, 5)}...`}
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>
                
                <Button 
                    variant="contained" 
                    onClick={startStream}
                    disabled={isStreaming}
                >
                    Start Stream
                </Button>
                <Button 
                    variant="contained" 
                    onClick={stopStream}
                    disabled={!isStreaming}
                    color="error"
                >
                    Stop Stream
                </Button>
            </Box>
            
            <video
                ref={videoRef}
                style={{ display: 'none' }}
                autoPlay
                playsInline
                muted
            />
            
            <canvas
                ref={canvasRef}
                style={{
                    border: '2px solid #333',
                    borderRadius: '8px',
                    maxWidth: '100%',
                    width: '640px',
                    height: '480px',
                    backgroundColor: '#000'
                }}
            />

            <Box sx={{ mt: 2 }}>
                <pre>
                    Frames received: {debugInfo.framesReceived}
                    Last frame: {debugInfo.lastFrameTime}
                </pre>
            </Box>
        </Box>
    );
};

export default SimpleWebcamTest;