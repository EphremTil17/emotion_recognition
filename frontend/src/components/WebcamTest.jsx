import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Box, Button, Modal, IconButton, Fade, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import CloseIcon from '@mui/icons-material/Close';

const WebcamTest = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const frameRequestRef = useRef(null);
    const lastFrameTime = useRef(0);
    const [open, setOpen] = useState(false);
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
        const FPS = 15;
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
        tempCanvas.width = 854;
        tempCanvas.height = 480;
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
            0.7
        );
    }, []);

    const startStream = useCallback(async () => {
        try {
            if (videoRef.current?.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            }
            if (frameRequestRef.current) {
                cancelAnimationFrame(frameRequestRef.current);
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: selectedCamera ? { exact: selectedCamera } : undefined,
                    frameRate: { ideal: 30 }
                }
            });
            
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
            
            const videoTrack = stream.getVideoTracks()[0];
            const settings = videoTrack.getSettings();
            const aspectRatio = settings.width / settings.height;
            const maxWidth =  854;  // Increased for better quality
            const maxHeight = 480;  // Increased for better quality
            let canvasWidth, canvasHeight;
            
            if (aspectRatio > maxWidth/maxHeight) {
                canvasWidth = maxWidth;
                canvasHeight = maxWidth / aspectRatio;
            } else {
                canvasHeight = maxHeight;
                canvasWidth = maxHeight * aspectRatio;
            }
            
            canvasRef.current.width = canvasWidth;
            canvasRef.current.height = canvasHeight;

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${import.meta.env.VITE_API_URL.replace(/^https?:\/\//, '')}/ws`;
            wsRef.current = new WebSocket(wsUrl);

            wsRef.current.onopen = () => {
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

    const handleCameraChange = (event) => {
        setSelectedCamera(event.target.value);
        if (isStreaming) {
            stopStream();
            setTimeout(() => startStream(), 100);
        }
    };

    const handleClose = () => {
        stopStream();
        setOpen(false);
    };

    useEffect(() => {
        return () => {
            if (frameRequestRef.current) {
                cancelAnimationFrame(frameRequestRef.current);
            }
            stopStream();
        };
    }, [stopStream]);

    return (
        <Box sx={{ mt: 3, textAlign: 'center' }}>
            <Button
                variant="contained"
                startIcon={<VideocamIcon sx={{ fontSize: '1.4rem' }} />}
                onClick={() => { setOpen(true); startStream(); }}
                sx={{
                    background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                    color: 'white',
                    px: 4,
                    py: 1.5,
                    fontFamily: 'Chakra Petch',
                    fontWeight: 600,
                    fontSize: '1.1rem',
                    letterSpacing: '0.5px',
                    textTransform: 'uppercase',
                    boxShadow: '0 4px 12px rgba(33, 150, 243, 0.2)',
                    transition: 'all 0.3s ease-in-out',
                    borderRadius: '8px',
                    '&:hover': {
                        background: 'linear-gradient(45deg, #1976D2 30%, #00BCD4 90%)',
                        boxShadow: '0 6px 16px rgba(33, 150, 243, 0.3)',
                        transform: 'translateY(-2px) scale(1.02)'
                    },
                    '&:active': {
                        transform: 'translateY(1px)',
                    }
                }}
            >
                Test Emotion Analyzer
            </Button>

            <Modal open={open} onClose={handleClose} closeAfterTransition>
                <Fade in={open}>
                    <Box sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        bgcolor: '#1a1a1a',
                        borderRadius: 3,
                        p: 4,
                        outline: 'none',
                        width: 'auto',
                        maxWidth: '95vw',
                        maxHeight: '95vh',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 2,
                    }}>
                        <Box sx={{ 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            mb: 2 
                        }}>
                            {cameras.length > 0 && (
                                <FormControl 
                                    sx={{ 
                                        flexGrow: 1,
                                        mr: 2,
                                        '& .MuiInputLabel-root': {
                                            color: 'white',
                                        },
                                        '& .MuiSelect-select': {
                                            color: 'white',
                                        },
                                        '& .MuiOutlinedInput-notchedOutline': {
                                            borderColor: 'rgba(255, 255, 255, 0.3)',
                                        },
                                        '&:hover .MuiOutlinedInput-notchedOutline': {
                                            borderColor: 'rgba(255, 255, 255, 0.5)',
                                        },
                                        '& .MuiSvgIcon-root': {
                                            color: 'white',
                                        }
                                    }}
                                >
                                    <InputLabel sx={{ color: 'white' }}>Camera</InputLabel>
                                    <Select
                                        value={selectedCamera}
                                        onChange={handleCameraChange}
                                        label="Camera"
                                        MenuProps={{
                                            PaperProps: {
                                                sx: {
                                                    bgcolor: '#2a2a2a',
                                                    color: 'white',
                                                    '& .MuiMenuItem-root': {
                                                        color: 'white',
                                                        '&:hover': {
                                                            bgcolor: 'rgba(255, 255, 255, 0.1)',
                                                        },
                                                    },
                                                },
                                            },
                                        }}
                                    >
                                        {cameras.map((camera) => (
                                            <MenuItem key={camera.deviceId} value={camera.deviceId}>
                                                {camera.label || `Camera ${camera.deviceId.slice(0, 5)}...`}
                                            </MenuItem>
                                        ))}
                                    </Select>
                                </FormControl>
                            )}
                            <IconButton
                                onClick={handleClose}
                                sx={{
                                    color: 'white',
                                    bgcolor: 'rgba(0,0,0,0.5)',
                                    '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' },
                                    transition: 'all 0.3s ease-in-out'
                                }}
                            >
                                <CloseIcon />
                            </IconButton>
                        </Box>

                        <Box sx={{ position: 'relative' }}>
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
                                    maxWidth: '100%',
                                    maxHeight: 'calc(95vh - 100px)',
                                    width: 'auto',
                                    height: 'auto',
                                    borderRadius: '12px',
                                    backgroundColor: '#000',
                                    objectFit: 'contain'
                                }}
                            />
                        </Box>
                    </Box>
                </Fade>
            </Modal>
        </Box>
    );
};

export default WebcamTest;