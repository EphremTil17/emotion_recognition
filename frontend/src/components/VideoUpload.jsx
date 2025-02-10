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
    padding: theme.spacing(6),
    textAlign: 'center',
    cursor: 'pointer',
    border: '2px dashed rgba(0, 0, 0, 0.1)',
    backgroundColor: 'rgba(0, 0, 0, 0.02)',
    transition: 'all 0.3s ease-in-out',
    '&:hover': {
        border: '2px dashed #2196F3',
        backgroundColor: 'rgba(33, 150, 243, 0.04)',
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
                        <Alert severity="error" sx={{ mb: 3, borderRadius: 2 }}>
                            {error}
                        </Alert>
                    )}
                    
                    {success && (
                        <Alert severity="success" sx={{ mb: 3, borderRadius: 2 }}>
                            Video uploaded and processed successfully!
                        </Alert>
                    )}

                    <UploadBox {...getRootProps()}>
                        <input {...getInputProps()} />
                        <CloudUploadIcon sx={{ 
                            fontSize: 64, 
                            color: isDragActive ? '#2196F3' : 'primary.main',
                            mb: 2,
                            transition: 'all 0.3s ease-in-out'
                        }} />
                        
                        {uploading ? (
                            <Box>
                                <CircularProgress size={30} sx={{ mb: 2 }} />
                                <Typography variant="h6" color="primary">
                                    Processing video...
                                </Typography>
                            </Box>
                        ) : (
                            <Typography variant="h6" color={isDragActive ? 'primary' : 'text.primary'}>
                                {isDragActive
                                    ? "Drop the video here..."
                                    : "Drag & drop a video here, or click to select"}
                            </Typography>
                        )}
                    </UploadBox>

                    <Box sx={{ mt: 3, textAlign: 'center' }}>
                        <Typography variant="body2" color="text.secondary" sx={{ 
                            p: 1,
                            borderRadius: 1,
                            backgroundColor: 'rgba(0, 0, 0, 0.04)'
                        }}>
                            Supported formats: MP4, AVI, MOV
                        </Typography>
                        
                        {success && (
                            <Button 
                                variant="contained" 
                                color="primary"
                                onClick={() => setShowResults(true)}
                                sx={{ mt: 2 }}
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