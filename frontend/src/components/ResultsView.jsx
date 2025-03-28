import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import Papa from 'papaparse';
import { 
    Box, 
    Button, 
    Typography, 
    Paper,
    Grid,
    CircularProgress,
    Divider,
    Select,
    MenuItem,
    FormControl,
    InputLabel
} from '@mui/material';

import {
    PieChart,
    Pie,
    Cell,
    Legend
} from 'recharts';

const COLORS = {
    Happy: '#05c793',    // Bright Green
    Sad: '#26547d',      // Medium Gray
    Neutral: '#fff5eb',  // Bright Blue
    Angry: '#ef436b',    // Bright Red
    Surprise: '#ffd033', // Bright Yellow   
    Undefined: '#dddddd' // Light Gray
};

const ENVIRONMENT_LABELS = {
    classroom: "Classroom",
    product_demo: "Product Demo",
    seminar: "Seminar",
    general: "General"
};

const CustomLegend = ({ payload }) => {
    return (
        <ul style={{ 
            listStyle: 'none', 
            padding: 0,
            display: 'flex',
            justifyContent: 'center',
            flexWrap: 'wrap',
            gap: '20px',
            marginTop: '20px'
        }}>
            {payload.map((entry, index) => (
                <li 
                    key={`legend-${index}`}
                    style={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                        fontSize: '13px',
                        color: '#666666'
                    }}
                >
                    <span style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '3px',
                        backgroundColor: entry.color,
                        marginRight: '8px',
                        display: 'inline-block'
                    }}/>
                    {entry.value}
                </li>
            ))}
        </ul>
    );
};

const CustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
    const RADIAN = Math.PI / 180;
    const radius = outerRadius * 1.2;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
    const sin = Math.sin(-midAngle * RADIAN);
    const cos = Math.cos(-midAngle * RADIAN);
    const textAnchor = cos >= 0 ? 'start' : 'end';

    return percent > 0.05 ? (
        <text 
            x={x}
            y={y}
            textAnchor={textAnchor}
            dominantBaseline="central"
            fontFamily="Inter, system-ui, -apple-system, sans-serif"
            fontSize="13"
            fontWeight="500"
            fill="#666666"
        >
            {`${(percent * 100).toFixed(1)}%`}
        </text>
    ) : null;
};

const StatsItem = ({ label, value }) => (
    <Box sx={{ mb: 1.5 }}>
        <Typography variant="body2" color="text.secondary" sx={{ 
            fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
            fontSize: '13px'
        }}>
            {label}
        </Typography>
        <Typography variant="body1" sx={{ 
            fontWeight: 500,
            fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
        }}>
            {value}
        </Typography>
    </Box>
);

const ContentWithEmotions = ({ timeSeriesData }) => {
    const [contentData, setContentData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [correlatedContent, setCorrelatedContent] = useState([]);

    useEffect(() => {
        fetch(`/api/content-analysis`, {
            method: 'GET',
            credentials: 'include',
            headers: {
                'Accept': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.text();
        })
        .then(text => {
            if (!text.trim()) {
                throw new Error("Empty response from server");
            }
            return JSON.parse(text);
        })
        .then(data => {
            setContentData(data);
            if (data && timeSeriesData) {
                const correlated = correlateEmotionsWithContent(timeSeriesData, data);
                setCorrelatedContent(correlated);
            }
            setLoading(false);
        })
        .catch(error => {
            console.error('Error fetching content analysis:', error);
            setLoading(false);
        });
    }, [timeSeriesData]);
  
    // Function to correlate emotion data with content
    const correlateEmotionsWithContent = (emotions, content) => {
      if (!content?.transcription?.results?.channels?.[0]?.alternatives?.[0]?.paragraphs?.paragraphs) {
        return [];
      }
      
      // Get paragraphs from content analysis
      const paragraphs = content.transcription.results.channels[0].alternatives[0].paragraphs.paragraphs;
      
      return paragraphs.map(paragraph => {
        // Find all emotions that occur during this paragraph
        const paragraphEmotions = emotions.filter(item => 
          item.timestamp >= paragraph.start && item.timestamp <= paragraph.end
        );
        
        // Calculate emotion distribution
        const emotionCounts = {};
        paragraphEmotions.forEach(item => {
          emotionCounts[item.emotion] = (emotionCounts[item.emotion] || 0) + 1;
        });
        
        const totalEmotions = paragraphEmotions.length;
        const emotionDistribution = Object.entries(emotionCounts).map(([emotion, count]) => ({
          emotion,
          percentage: totalEmotions > 0 ? (count / totalEmotions) * 100 : 0
        }));
        
        // Find dominant emotion
        const dominantEmotion = emotionDistribution.length > 0 
          ? emotionDistribution.sort((a, b) => b.percentage - a.percentage)[0].emotion 
          : 'Neutral';
        
        // Calculate average engagement score
        const avgEngagement = paragraphEmotions.length > 0 
          ? paragraphEmotions.reduce((sum, item) => sum + parseFloat(item.general_score || 0), 0) / paragraphEmotions.length
          : 0;
        
        return {
          text: paragraph.sentences?.map(s => s.text).join(' ') || '',
          start: paragraph.start,
          end: paragraph.end,
          dominantEmotion,
          emotionDistribution,
          avgEngagement: parseFloat(avgEngagement.toFixed(2))
        };
      });
    };
  
    // Format time in MM:SS format
    const formatTime = (seconds) => {
      const mins = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      return `${mins}:${secs.toString().padStart(2, '0')}`;
    };
  
    if (loading) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
          <CircularProgress size={24} />
        </Box>
      );
    }
  
    if (!contentData?.transcription) {
      return (
        <Typography variant="body1" sx={{ my: 2, color: 'text.secondary' }}>
          No content analysis available. The video may not contain speech or the analysis is still processing.
        </Typography>
      );
    }
  
    // Display the summary
    const summary = contentData.transcription?.results?.summary?.short;
  
    return (
      <Box sx={{ mt: 2 }}>
        {summary && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle1" fontWeight={500} sx={{
              fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
            }}>
              Summary
            </Typography>
            <Paper variant="outlined" sx={{ p: 2, backgroundColor: '#f8f9fa' }}>
              <Typography variant="body1" sx={{
                fontStyle: 'italic',
                fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                fontSize: '14px'
              }}>
                {summary}
              </Typography>
            </Paper>
          </Box>
        )}
  
        <Typography variant="subtitle1" fontWeight={500} sx={{
          fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
          mb: 2
        }}>
          Topics with Emotional Response
        </Typography>
  
        {correlatedContent.map((item, index) => (
          <Paper 
            key={index} 
            variant="outlined" 
            sx={{ 
              p: 2, 
              mb: 2, 
              borderLeft: `4px solid ${COLORS[item.dominantEmotion] || '#ccc'}`,
              transition: 'all 0.2s ease',
              '&:hover': {
                boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
              }
            }}
          >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="caption" sx={{ 
                color: 'text.secondary',
                fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
              }}>
                {formatTime(item.start)} - {formatTime(item.end)}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ 
                  width: 10, 
                  height: 10, 
                  borderRadius: '50%', 
                  backgroundColor: COLORS[item.dominantEmotion] || '#ccc' 
                }} />
                <Typography variant="caption" sx={{ 
                  fontWeight: 500,
                  fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                }}>
                  {item.dominantEmotion} | Engagement: {item.avgEngagement.toFixed(1)}
                </Typography>
              </Box>
            </Box>
            <Typography variant="body2" sx={{ 
              fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
            }}>
              {item.text}
            </Typography>
          </Paper>
        ))}
      </Box>
    );
  };

const ScholarlyResources = () => {
    const [scholarlyData, setScholarlyData] = useState(null);
    const [loading, setLoading] = useState(true);
  
    useEffect(() => {
      fetch(`/api/scholarly-results`, {
        method: 'GET',
        credentials: 'include',
        headers: {
          'Accept': 'application/json'
        }
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        setScholarlyData(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error fetching scholarly results:', error);
        setLoading(false);
      });
    }, []);
  
    if (loading) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
          <CircularProgress size={24} />
        </Box>
      );
    }
  
    if (!scholarlyData || scholarlyData.error) {
      return (
        <Typography variant="body1" sx={{ my: 2, color: 'text.secondary' }}>
          No scholarly resources available. Try processing a video with speech content.
        </Typography>
      );
    }
  
    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="subtitle1" fontWeight={500} sx={{
          fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
          mb: 2
        }}>
          Scholarly Resources
        </Typography>
        
        <Typography variant="caption" sx={{ 
          color: 'text.secondary',
          fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
          display: 'block',
          mb: 2
        }}>
          Search query: {scholarlyData.optimized_query}
        </Typography>
  
        {scholarlyData.results.map((item, index) => (
          <Paper 
            key={index} 
            variant="outlined" 
            sx={{ 
              p: 2, 
              mb: 2, 
              transition: 'all 0.2s ease',
              '&:hover': {
                boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
              }
            }}
          >
            <Typography variant="subtitle1" sx={{ 
              fontWeight: 500,
              fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
            }}>
              {item.title}
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ 
              mb: 1,
              fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
            }}>
              {item.authors} ({item.year}) - {item.venue}
            </Typography>
            
            <Typography variant="body2" sx={{ 
              mb: 2,
              fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
            }}>
              {item.abstract.length > 250 ? `${item.abstract.substring(0, 250)}...` : item.abstract}
            </Typography>
            
            {item.url && (
              <Button 
                variant="outlined" 
                size="small"
                href={item.url} 
                target="_blank"
                sx={{ 
                  fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                  textTransform: 'none'
                }}
              >
                View Resource
              </Button>
            )}
          </Paper>
        ))}
      </Box>
    );
  };

const ResultsView = ({ processedResults, onClose }) => {
    const [selectedEnvironment, setSelectedEnvironment] = useState('classroom');
    const [timeSeriesData, setTimeSeriesData] = useState(null);
    const [averageType, setAverageType] = useState('30s');

    useEffect(() => {
        fetch('/api/analytics', {
            method: 'GET',
            headers: {
                'Accept': 'text/csv',
            },
            credentials: 'include'
        })
        .then(response => response.text())
        .then(csv => {
            const results = Papa.parse(csv, { 
                header: true, 
                dynamicTyping: true,
                skipEmptyLines: true 
            });
            if (results.data && results.data.length > 0) {
                setTimeSeriesData(results.data);
            }
        })
        .catch(error => console.error('Error fetching analytics:', error));
    }, []);

    if (!processedResults) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
                <CircularProgress />
            </Box>
        );
    }

    const currentScore = processedResults.engagement_scores?.[selectedEnvironment] || 'N/A';

    const distributionData = Object.entries(processedResults.emotion_distribution)
        .filter(([name]) => name !== 'Undefined' || processedResults.emotion_distribution[name] > 0)
        .map(([name, value]) => ({ 
            name, 
            value: parseFloat(value.toFixed(1))
        }));

    const chartOptions = {
        chart: {
            type: 'area',
            height: 350,
            toolbar: {
                show: true
            },
            fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
        },
        dataLabels: {
            enabled: false
        },
        stroke: {
            curve: 'smooth',
            width: 2
        },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.7,
                opacityTo: 0.3,
            }
        },
        xaxis: {
            type: 'numeric',
            title: {
                text: 'Time (seconds)',
                style: {
                    fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                }
            },
            labels: {
                formatter: (value) => `${value}s`,
                style: {
                    fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                }
            }
        },
        yaxis: {
            min: 1,
            max: 5,
            title: {
                text: 'Engagement Score',
                style: {
                    fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                }
            }
        },
        tooltip: {
            x: {
                formatter: (value) => `${value}s`
            },
            style: {
                fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
            }
        }
    };

    const getChartSeries = () => {
        if (!timeSeriesData) return [];

        const columnMap = {
            '30s': `${selectedEnvironment}_30s_avg`,
            '1min': `${selectedEnvironment}_1min_avg`,
            'cumulative': `${selectedEnvironment}_cumulative`
        };

        const column = columnMap[averageType];
        
        return [{
            name: 'Engagement Score',
            data: timeSeriesData
                .filter(row => row && row.timestamp !== undefined)
                .map(row => ({
                    x: parseFloat(row.timestamp),
                    y: parseFloat(row[column])
                }))
                .filter(point => !isNaN(point.y))
        }];
    };

    return (
        <Box sx={{ mt: 4 }}>
            <Typography variant="h4" gutterBottom sx={{ 
                fontWeight: 300,
                fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
            }}>
                Analysis Results
            </Typography>
            
            <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 3, height: '100%' }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography variant="h6" gutterBottom sx={{
                                fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                            }}>
                                Engagement Score
                            </Typography>
                            <FormControl size="small" sx={{ minWidth: 150 }}>
                                <InputLabel id="environment-select-label">Environment</InputLabel>
                                <Select
                                    labelId="environment-select-label"
                                    value={selectedEnvironment}
                                    label="Environment"
                                    onChange={(e) => setSelectedEnvironment(e.target.value)}
                                    sx={{ 
                                        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                                        fontSize: '14px'
                                    }}
                                >
                                    {Object.entries(ENVIRONMENT_LABELS).map(([key, label]) => (
                                        <MenuItem 
                                            key={key} 
                                            value={key}
                                            sx={{ 
                                                fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                                                fontSize: '14px'
                                            }}
                                        >
                                            {label}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Box>
                        <Typography variant="h2" color="primary" sx={{ 
                            mb: 3, 
                            fontWeight: 300,
                            fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                        }}>
                            {currentScore.toFixed(1)}/5.0
                        </Typography>
                        
                        <Divider sx={{ my: 2 }} />
                        
                        <Typography variant="h6" gutterBottom sx={{ 
                            mt: 2,
                            fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                        }}>
                            Processing Statistics
                        </Typography>
                        
                        <StatsItem 
                            label="Frames Processed"
                            value={processedResults.frames_processed}
                        />
                        <StatsItem 
                            label="Face Detection Rate"
                            value={`${processedResults.face_detection_rate.toFixed(1)}%`}
                        />
                        <StatsItem 
                            label="Duration"
                            value={`${(processedResults.duration_seconds).toFixed(1)} seconds`}
                        />
                    </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 3, height: '100%' }}>
                        <Typography variant="h6" gutterBottom sx={{
                            fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                        }}>
                            Emotion Distribution
                        </Typography>
                        <Box sx={{ 
                            display: 'flex', 
                            justifyContent: 'center', 
                            alignItems: 'center', 
                            height: 'calc(100% - 32px)'
                        }}>
                            <PieChart width={500} height={400}>
                                <Pie
                                    data={distributionData}
                                    dataKey="value"
                                    nameKey="name"
                                    cx="50%"
                                    cy="45%"
                                    outerRadius={130}
                                    startAngle={-270}
                                    endAngle={90}
                                    labelLine={false}
                                    label={CustomLabel}
                                >
                                    {distributionData.map((entry) => (
                                        <Cell 
                                            key={entry.name} 
                                            fill={COLORS[entry.name]}
                                            stroke="none"
                                        />
                                    ))}
                                </Pie>
                                <Legend 
                                    content={CustomLegend}
                                    verticalAlign="bottom"
                                    align="center"
                                />
                            </PieChart>
                        </Box>
                    </Paper>
                </Grid>
            </Grid>

            <Grid container spacing={3} sx={{ mt: 2 }}>
                <Grid item xs={12}>
                    <Paper sx={{ p: 3 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography variant="h6" sx={{
                                fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                            }}>
                                Engagement Over Time
                            </Typography>
                            <FormControl size="small" sx={{ minWidth: 150 }}>
                                <InputLabel>Average Type</InputLabel>
                                <Select
                                    value={averageType}
                                    label="Average Type"
                                    onChange={(e) => setAverageType(e.target.value)}
                                    sx={{ 
                                        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                                        fontSize: '14px'
                                    }}
                                >
                                    <MenuItem value="30s">30 Second Average</MenuItem>
                                    <MenuItem value="1min">1 Minute Average</MenuItem>
                                    <MenuItem value="cumulative">Cumulative Average</MenuItem>
                                </Select>
                            </FormControl>
                        </Box>
                        {timeSeriesData && (
                            <ReactApexChart
                                options={chartOptions}
                                series={getChartSeries()}
                                type="area"
                                height={350}
                            />
                        )}
                    </Paper>
                </Grid>
            </Grid>

            <Grid container spacing={3} sx={{ mt: 2 }}>
                <Grid item xs={12}>
                    <Paper sx={{ p: 3 }}>
                        <Typography variant="h6" sx={{
                            fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                        }}>
                        Content Analysis
                    </Typography>
                    <ContentWithEmotions timeSeriesData={timeSeriesData} />
                    </Paper>
                </Grid>
            </Grid>

            <Grid container spacing={3} sx={{ mt: 2 }}>
                <Grid item xs={12}>
                    <Paper sx={{ p: 3 }}>
                        <Typography variant="h6" sx={{
                            fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                        }}>
                            Related Resources
                        </Typography>
                        <ScholarlyResources />
                    </Paper>
                </Grid>
            </Grid>

            <Box sx={{ 
                mt: 3, 
                textAlign: 'center',
                display: 'flex',
                justifyContent: 'center'
            }}>
                <Button
                    variant="contained"
                    onClick={onClose}
                    sx={{
                        px: 4,
                        py: 1.5,
                        fontFamily: 'Lexend',
                        fontWeight: 600,
                        fontSize: '1.1rem',
                        letterSpacing: '0.5px',
                        textTransform: 'capitalize',  // Makes it match the other buttons
                        boxShadow: '0 4px 12px rgba(33, 150, 243, 0.2)',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                            boxShadow: '0 6px 16px rgba(33, 150, 243, 0.3)',
                            transform: 'translateY(-1px)'
                        }
                    }}
                >
                    Close Results
                </Button>
            </Box>
        </Box>
    );
};

export default ResultsView;