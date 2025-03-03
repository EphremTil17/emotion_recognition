import React from 'react';
import { AppBar, Toolbar, Typography, IconButton, Box } from '@mui/material';
import EmojiEmotionsIcon from '@mui/icons-material/EmojiEmotions';
import GitHubIcon from '@mui/icons-material/GitHub';

const Header = () => {
    return (
        <AppBar position="static" sx={{ marginBottom: 2, background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)' }}>
            <Toolbar>
                <EmojiEmotionsIcon sx={{ mr: 2 }} />
                <Typography variant="h5" sx={{ 
                    flexGrow: 1,
                    fontFamily: 'Chakra Petch',
                    fontWeight: 600,
                }}>
                    Engage AI
                </Typography>
                <IconButton
                    color="inherit"
                    href="https://github.com/EphremTil17/emotion_recognition/tree/mainline"
                    target="_blank"
                    sx={{ ml: 2 }}
                >
                    <GitHubIcon />
                </IconButton>
            </Toolbar>
        </AppBar>
    );
};

export default Header;