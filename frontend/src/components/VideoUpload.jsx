import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import ResultsView from './ResultsView';
import { 
    Box, 
    Button, 
    CircularProgress, 
    Typography,
    Paper,
    Alert
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { styled } from '@mui/material/styles';

const UploadBox = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(8), // Increased padding
    textAlign: 'center',
    cursor: 'pointer',
    border: '2px dashed rgba(0, 0, 0, 0.1)',
    backgroundColor: 'rgba(0, 0, 0, 0.02)',
    transition: 'all 0.3s ease-in-out',
    borderRadius: '16px', // More rounded corners
    '&:hover': {
        border: '2px dashed #2196F3',
        backgroundColor: 'rgba(33, 150, 243, 0.04)',
        transform: 'translateY(-5px)', // Subtle lift effect
        boxShadow: '0 8px 20px rgba(0, 0, 0, 0.1)',
    },
}));

const VideoUpload = () => {
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(false);
    const [progress, setProgress] = useState(0);
    const [processedResults, setProcessedResults] = useState(null);
    const [showResults, setShowResults] = useState(false);
    
    const onDrop = useCallback(async (acceptedFiles) => {
        const file = acceptedFiles[0];
        if (!file) return;

        setUploading(true);
        setError(null);
        setSuccess(false);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            const data = await response.json();
            setProcessedResults(data);
            setSuccess(true);
            console.log('Upload successful:', data);
        } catch (err) {
            setError(err.message);
            console.error('Upload error:', err);
        } finally {
            setUploading(false);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'video/*': ['.mp4', '.avi', '.mov']
        },
        maxFiles: 1
    });

    return (
        <Box>
            {showResults && processedResults ? (
                <ResultsView 
                    processedResults={processedResults}
                    onClose={() => setShowResults(false)}
                />
            ) : (
                <>
                    {error && (
                        <Alert 
                            severity="error" 
                            sx={{ 
                                mb: 3, 
                                borderRadius: 2,
                                fontFamily: 'Lexend',
                                '& .MuiAlert-message': {
                                    fontWeight: 500
                                }
                            }}
                        >
                            {error}
                        </Alert>
                    )}

                    {success && (
                        <Alert 
                            severity="success" 
                            sx={{ 
                                mb: 3, 
                                borderRadius: 2,
                                fontFamily: 'Lexend',
                                '& .MuiAlert-message': {
                                    fontWeight: 500
                                }
                            }}
                        >
                            Video uploaded and processed successfully!
                        </Alert>
                    )}

                    <UploadBox {...getRootProps()}>
                        <input {...getInputProps()} />
                        <CloudUploadIcon sx={{ 
                            fontSize: 80, // Larger icon
                            color: isDragActive ? '#2196F3' : 'primary.main',
                            mb: 3,
                            transition: 'all 0.3s ease-in-out',
                            transform: isDragActive ? 'scale(1.1)' : 'scale(1.5)', // Add scale effect
                        }} />
                        
                        {uploading ? (
                            <Box>
                                <CircularProgress size={40} sx={{ mb: 3 }} />
                                <Typography 
                                    variant="h6" 
                                    color="primary"
                                    sx={{ 
                                        fontFamily: 'Chakra Petch',
                                        fontWeight: 600,
                                        letterSpacing: '0.5px'
                                    }}
                                >
                                    Processing video...
                                </Typography>
                            </Box>
                        ) : (
                            <Typography 
                                variant="h5" 
                                sx={{ 
                                    color: isDragActive ? 'primary.main' : 'text.primary',
                                    fontFamily: 'Chakra Petch',
                                    fontWeight: 500,
                                    letterSpacing: '0.5px',
                                    mb: 2,
                                    opacity: 0.8
                                }}
                            >
                                {isDragActive
                                    ? "Drop the video here..."
                                    : "Drag & drop a video here, or click to select"}
                            </Typography>
                        )}

                    </UploadBox>

                    <Box sx={{ 
                        mt: 3, 
                        textAlign: 'center',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: 2  // Creates consistent spacing between elements
                    }}>
                        <Typography 
                            variant="body2" 
                            color="text.secondary" 
                            sx={{ 
                                p: 1.5,
                                borderRadius: 2,
                                backgroundColor: 'rgba(0, 0, 0, 0.04)',
                                fontFamily: 'Lexend',
                                fontSize: '0.875rem',
                                fontWeight: 400,
                                display: 'inline-block',
                                border: '1px solid rgba(0, 0, 0, 0.08)',
                                width: 'auto'  // Allows the box to size to content
                            }}
                        >
                            Supported formats: MP4, AVI, MOV
                        </Typography>
                        
                        {success && (
                            <Button 
                                variant="contained" 
                                color="primary"
                                onClick={() => setShowResults(true)}
                                sx={{ 
                                    px: 4,
                                    py: 1.5,
                                    fontFamily: 'Lexend',
                                    textTransform: 'capitalize', 
                                    fontWeight: 600,
                                    fontSize: '1.1rem',
                                    letterSpacing: '0.5px',
                                    boxShadow: '0 4px 12px rgba(33, 150, 243, 0.2)',
                                    '&:hover': {
                                        boxShadow: '0 6px 16px rgba(33, 150, 243, 0.3)',
                                        transform: 'translateY(-1px)'
                                    }
                                }}
                            >
                                View Analysis Results
                            </Button>
                        )}
                    </Box>
                </>
            )}
        </Box>
    );
};

export default VideoUpload;