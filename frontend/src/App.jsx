import React from 'react'
import { Box, Container, Typography, Paper } from '@mui/material'
import VideoUpload from './components/VideoUpload'
import WebcamTest from './components/WebcamTest'
import Header from './components/Header'
import backgroundImage from './assets/images/background.png'

function App() {
  return (
    <>
      <Header />
      <Container maxWidth="lg">
        <Box sx={{ my: 6 }}>
          {/* Changed to vertical stack with larger image */}
          <Box sx={{ 
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            // gap: 2,
          }}>
            <Box sx={{
              width: '250px',
              height: '250px',
              backgroundImage: `url(${backgroundImage})`,
              backgroundSize: 'contain',
              backgroundPosition: 'center',
              backgroundRepeat: 'no-repeat',
              transform: 'scale(1.5)', // Makes it 20% larger
              mb: 1,
              mt: 4,
            }} />
            <Typography variant="h3" component="h1" gutterBottom sx={{
              fontFamily: 'Chakra Petch',
              fontWeight: 700,
              background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              color: 'transparent',
            }}>
              Emotion Recognition
            </Typography>
          </Box>

          <Typography 
            variant="h6" 
            align="center" 
            sx={{ 
              mb: 6,
              maxWidth: '800px',
              mx: 'auto',
              color: '#666',
              lineHeight: 1.6,
              fontFamily: 'Chakra Petch',
              fontWeight: 400,
            }}
          >
            Transform your video lectures into valuable insights. Our AI-powered system analyzes audience 
            facial expressions in real-time, providing detailed engagement metrics and emotional response data 
            to help you optimize your content delivery.
          </Typography>

          <Paper 
            elevation={3} 
            sx={{ 
              p: 6,
              backgroundColor: 'rgba(255, 255, 255, 0.98)',
              borderRadius: '16px',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
              position: 'relative',
              overflow: 'hidden',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '4px',
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
              }
            }}
          >
            <VideoUpload />
            <WebcamTest />
            <Box sx={{ 
              mt: 4, 
              pt: 3, 
              borderTop: '1px solid #eee',
              textAlign: 'center' 
            }}>
              <Typography 
                variant="body2" 
                color="text.secondary"
                sx={{ 
                  fontFamily: 'Inter',
                  fontSize: '0.875rem' 
                }}
              >
                <a 
                  href="https://github.com/EphremTil17/emotion_recognition/tree/mainline" 
                  style={{ 
                    color: '#2196F3',
                    textDecoration: 'none'
                  }}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  View on GitHub
                </a>
              </Typography>
            </Box>
          </Paper>
        </Box>
      </Container>
    </>
  );
}

export default App