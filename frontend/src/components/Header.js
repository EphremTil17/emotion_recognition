import React from 'react';
import { AppBar, Toolbar, Typography, Box } from '@mui/material';
import EmojiEmotionsIcon from '@mui/icons-material/EmojiEmotions';

const Header = () => {
    return (
        <AppBar position="static">
            <Toolbar>
                <EmojiEmotionsIcon sx={{ mr: 2 }} />
                <Typography variant="h6" component="div">
                    Engage AI - Emotion Recognition
                </Typography>
                <Box sx={{ flexGrow: 1 }} />
            </Toolbar>
        </AppBar>
    );
};

export default Header;