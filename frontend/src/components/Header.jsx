import React from 'react';
import { AppBar, Toolbar, Typography } from '@mui/material';
import EmojiEmotionsIcon from '@mui/icons-material/EmojiEmotions';

const Header = () => {
    return (
        <AppBar position="static" sx={{ marginBottom: 2 }}>
            <Toolbar>
                <EmojiEmotionsIcon sx={{ mr: 2 }} />
                <Typography variant="h6">
                    Emotion Recognition
                </Typography>
            </Toolbar>
        </AppBar>
    );
};

export default Header;