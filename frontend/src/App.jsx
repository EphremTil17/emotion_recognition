import React from 'react'
import { Box, Container, Typography, Paper } from '@mui/material'
import VideoUpload from './components/VideoUpload'
import WebcamTest from './components/WebcamTest'
import Header from './components/Header'

function App() {
  return (
    <>
      <Header />
      <Container maxWidth="lg">
        <Box sx={{ my: 6 }}>
          <Typography variant="h3" component="h1" gutterBottom align="center" sx={{
            fontWeight: 600,
            background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            color: 'transparent'
          }}>
            Emotion Recognition
          </Typography>
          <Typography variant="h6" align="center" color="text.secondary" sx={{ mb: 4 }}>
            Upload a recorded Zoom lecture video to analyze target audience engagement levels. The system will process the video and generate an engagement report for you.
          </Typography>
          <Paper elevation={3} sx={{ p: 4, backgroundColor: 'rgba(255, 255, 255, 0.8)' }}>
            <VideoUpload />
            <WebcamTest />
          </Paper>
        </Box>
      </Container>
    </>
  );
}

export default App