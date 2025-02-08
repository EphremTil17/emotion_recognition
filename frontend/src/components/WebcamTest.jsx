import React, { useRef, useState, useCallback } from 'react';
import { Box, Button, Modal, IconButton, Fade } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import CloseIcon from '@mui/icons-material/Close';

const WebcamTest = () => {
    const videoRef = useRef(null);
    const streamRef = useRef(null);
    const [open, setOpen] = useState(false);
    const [error, setError] = useState(null);

    const startWebcam = useCallback(async () => {
        try {
            setError(null);
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user' },
                audio: false
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
                streamRef.current = stream;
            }
        } catch (err) {
            setError(`Camera access denied. Please check your permissions.`);
        }
    }, []);

    const handleClose = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        setOpen(false);
        setError(null);
    };

    return (
        <Box sx={{ mt: 3, textAlign: 'center' }}>
            <Button
                variant="contained"
                startIcon={<VideocamIcon />}
                onClick={() => { setOpen(true); startWebcam(); }}
                sx={{
                    background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                    color: 'white',
                    px: 4,
                    py: 1.5,
                    fontSize: '1.1rem',
                    transition: 'all 0.3s ease-in-out',
                    '&:hover': {
                        background: 'linear-gradient(45deg, #1976D2 30%, #00BCD4 90%)',
                        transform: 'scale(1.05)'
                    }
                }}
            >
                Test Emotion Analyzer
            </Button>

            <Modal
                open={open}
                onClose={handleClose}
                closeAfterTransition
            >
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
                    }}>
                        <IconButton
                            onClick={handleClose}
                            sx={{
                                position: 'absolute',
                                right: 16,
                                top: 16,
                                color: 'white',
                                bgcolor: 'rgba(0,0,0,0.5)',
                                '&:hover': {
                                    bgcolor: 'rgba(255,255,255,0.2)',
                                },
                                transition: 'all 0.3s ease-in-out'
                            }}
                        >
                            <CloseIcon />
                        </IconButton>

                        {error ? (
                            <Box sx={{ color: 'error.main', p: 2 }}>{error}</Box>
                        ) : (
                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                muted
                                style={{
                                    maxWidth: '90vw',
                                    maxHeight: '80vh',
                                    borderRadius: '12px'
                                }}
                            />
                        )}
                    </Box>
                </Fade>
            </Modal>
        </Box>
    );
};

export default WebcamTest;