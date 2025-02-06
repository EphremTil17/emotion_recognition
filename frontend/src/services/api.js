import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

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
        const response = await api.post('/process-video', formData, {
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
        throw new Error(error.response?.data?.message || 'Error uploading video');
    }
};

export const checkProcessingStatus = async (jobId) => {
    try {
        const response = await api.get(`/status/${jobId}`);
        return response.data;
    } catch (error) {
        throw new Error(error.response?.data?.message || 'Error checking status');
    }
};

export const getProcessedVideo = async (videoId) => {
    try {
        const response = await api.get(`/download/${videoId}`);
        return response.data;
    } catch (error) {
        throw new Error(error.response?.data?.message || 'Error getting video');
    }
};

export default api;