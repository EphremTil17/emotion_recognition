import axios from 'axios';

// Use Vite's environment variable naming convention
const API_URL = import.meta.env.VITE_API_URL || 'http://backend:8001';

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const uploadVideo = async (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await api.post('/process', formData, {  // Changed endpoint to match backend
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: (progressEvent) => {
                const percentCompleted = Math.round(
                    (progressEvent.loaded * 100) / progressEvent.total
                );
                onProgress && onProgress(percentCompleted);
            },
        });
        return response.data;
    } catch (error) {
        console.error('Upload error:', error);
        throw new Error(error.response?.data?.error || 'Error uploading video');
    }
};

export const checkProcessingStatus = async (jobId) => {
    try {
        const response = await api.get(`/health`);  // Changed to use health endpoint
        return response.data;
    } catch (error) {
        console.error('Status check error:', error);
        throw new Error(error.response?.data?.error || 'Error checking status');
    }
};

export const getProcessedVideo = async (videoPath) => {
    try {
        // Assuming the processed video path is returned from the upload response
        const response = await api.get(`/download`, {
            params: { path: videoPath },
            responseType: 'blob'  // Important for handling video files
        });
        return response.data;
    } catch (error) {
        console.error('Download error:', error);
        throw new Error(error.response?.data?.error || 'Error getting video');
    }
};

// Add error interceptor for better error handling
api.interceptors.response.use(
    response => response,
    error => {
        console.error('API Error:', error);
        if (error.response?.status === 404) {
            throw new Error('Resource not found');
        }
        if (error.response?.status === 500) {
            throw new Error('Server error');
        }
        throw error;
    }
);

export default api;