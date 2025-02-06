import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
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
    padding: theme.spacing(3),
    textAlign: 'center',
    cursor: 'pointer',
    border: '2px dashed #ccc',
    '&:hover': {
        border: '2px dashed #999',
    },
}));

const VideoUpload = () => {
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(false);
    const [progress, setProgress] = useState(0);

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
            setSuccess(true);
            // Handle successful upload (e.g., show download link)
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
        <Box sx={{ maxWidth: 600, mx: 'auto', mt: 4 }}>
            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}
            
            {success && (
                <Alert severity="success" sx={{ mb: 2 }}>
                    Video uploaded and processed successfully!
                </Alert>
            )}

            <UploadBox {...getRootProps()}>
                <input {...getInputProps()} />
                <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                
                {uploading ? (
                    <Box>
                        <CircularProgress size={24} sx={{ mb: 1 }} />
                        <Typography>Processing video...</Typography>
                    </Box>
                ) : (
                    <Typography>
                        {isDragActive
                            ? "Drop the video here..."
                            : "Drag & drop a video here, or click to select"}
                    </Typography>
                )}
            </UploadBox>

            <Box sx={{ mt: 2, textAlign: 'center' }}>
                <Typography variant="caption" color="textSecondary">
                    Supported formats: MP4, AVI, MOV
                </Typography>
            </Box>
        </Box>
    );
};

export default VideoUpload;