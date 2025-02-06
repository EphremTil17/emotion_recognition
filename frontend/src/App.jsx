import React from 'react'
import { Box, Container, Typography } from '@mui/material'
import VideoUpload from './components/VideoUpload'

function App() {
  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          Emotion Recognition
        </Typography>
        <VideoUpload />
      </Box>
    </Container>
  )
}

export default App